"""Azure Entra ID authentication integration for Streamlit."""

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import streamlit as st

from skplay.config import settings

# MSAL is optional - only import if auth is enabled
try:
    import msal

    MSAL_AVAILABLE = True
except ImportError:
    MSAL_AVAILABLE = False
    msal = None


@dataclass
class AuthenticatedUser:
    """Represents an authenticated user from Azure Entra ID."""

    oid: str  # Azure Object ID
    email: str
    display_name: str
    given_name: str | None = None
    family_name: str | None = None
    roles: list[str] | None = None
    groups: list[str] | None = None
    access_token: str | None = None
    token_expiry: datetime | None = None

    @classmethod
    def from_claims(cls, claims: dict[str, Any], access_token: str | None = None) -> "AuthenticatedUser":
        """Create AuthenticatedUser from ID token claims."""
        return cls(
            oid=claims.get("oid", ""),
            email=claims.get("preferred_username", claims.get("email", "")),
            display_name=claims.get("name", ""),
            given_name=claims.get("given_name"),
            family_name=claims.get("family_name"),
            roles=claims.get("roles", []),
            groups=claims.get("groups", []),
            access_token=access_token,
            token_expiry=datetime.utcnow() + timedelta(hours=1) if access_token else None,
        )


class AzureEntraAuth:
    """Azure Entra ID authentication handler using MSAL."""

    AUTHORITY_URL = "https://login.microsoftonline.com/{tenant_id}"
    SCOPES = ["User.Read"]

    def __init__(self):
        if not MSAL_AVAILABLE:
            raise RuntimeError("msal package not installed. Install with: pip install msal")

        self.tenant_id = settings.azure_tenant_id
        self.client_id = settings.azure_client_id
        self.client_secret = settings.azure_client_secret.get_secret_value()
        self.redirect_uri = settings.azure_redirect_uri
        self.authority = self.AUTHORITY_URL.format(tenant_id=self.tenant_id)

        self._msal_app = msal.ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=self.authority,
        )

    def get_auth_url(self, state: str | None = None) -> str:
        """Generate the authorization URL for the login flow."""
        state = state or secrets.token_urlsafe(32)

        # Store state in session for CSRF validation
        st.session_state["auth_state"] = state

        auth_url = self._msal_app.get_authorization_request_url(
            scopes=self.SCOPES,
            state=state,
            redirect_uri=self.redirect_uri,
        )
        return auth_url

    def handle_callback(self, code: str, state: str) -> AuthenticatedUser | None:
        """Handle the OAuth callback and exchange code for tokens."""
        # Validate state (CSRF protection)
        stored_state = st.session_state.get("auth_state")
        if state != stored_state:
            st.error("Invalid authentication state. Please try logging in again.")
            return None

        # Exchange code for tokens
        result = self._msal_app.acquire_token_by_authorization_code(
            code=code,
            scopes=self.SCOPES,
            redirect_uri=self.redirect_uri,
        )

        if "error" in result:
            st.error(f"Authentication error: {result.get('error_description', result.get('error'))}")
            return None

        # Extract user info from ID token claims
        id_token_claims = result.get("id_token_claims", {})
        access_token = result.get("access_token")

        user = AuthenticatedUser.from_claims(id_token_claims, access_token)

        # Clear auth state
        st.session_state.pop("auth_state", None)

        return user

    def logout_url(self, post_logout_redirect_uri: str | None = None) -> str:
        """Generate the logout URL."""
        params = {"client_id": self.client_id}
        if post_logout_redirect_uri:
            params["post_logout_redirect_uri"] = post_logout_redirect_uri

        return f"{self.authority}/oauth2/v2.0/logout?{urlencode(params)}"


def init_auth_session() -> None:
    """Initialize authentication-related session state."""
    if "authenticated_user" not in st.session_state:
        st.session_state.authenticated_user = None
    if "auth_state" not in st.session_state:
        st.session_state.auth_state = None


def get_current_user() -> AuthenticatedUser | None:
    """Get the currently authenticated user from session state."""
    return st.session_state.get("authenticated_user")


def set_current_user(user: AuthenticatedUser | None) -> None:
    """Set the current user in session state."""
    st.session_state.authenticated_user = user


def is_authenticated() -> bool:
    """Check if a user is currently authenticated."""
    user = get_current_user()
    if not user:
        return False

    # Check token expiry
    if user.token_expiry and datetime.utcnow() > user.token_expiry:
        set_current_user(None)
        return False

    return True


def require_auth(func):
    """Decorator to require authentication for a page/function."""

    def wrapper(*args, **kwargs):
        if not settings.enable_auth:
            return func(*args, **kwargs)

        if not is_authenticated():
            show_login_prompt()
            st.stop()

        return func(*args, **kwargs)

    return wrapper


def show_login_prompt() -> None:
    """Show login prompt UI."""
    st.warning("Please log in to access this feature.")

    if not settings.is_azure_auth_configured:
        st.error("Authentication is not configured. Contact administrator.")
        return

    try:
        auth = AzureEntraAuth()
        auth_url = auth.get_auth_url()

        st.markdown(f"""
        <a href="{auth_url}" target="_self">
            <button style="
                background-color: #0078d4;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
            ">
                Sign in with Microsoft
            </button>
        </a>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Authentication error: {e}")


def show_user_info() -> None:
    """Show current user info in sidebar."""
    user = get_current_user()
    if not user:
        return

    with st.sidebar:
        st.markdown("---")
        st.markdown(f"**{user.display_name}**")
        st.caption(user.email)

        if st.button("Sign Out", key="signout_btn"):
            set_current_user(None)
            st.rerun()


def render_auth_callback_handler() -> bool:
    """Handle OAuth callback - call this at the top of your app.

    Returns:
        True if callback was handled, False otherwise.
    """
    # Check for OAuth callback parameters
    query_params = st.query_params

    code = query_params.get("code")
    state = query_params.get("state")

    if not code or not state:
        return False

    if not settings.is_azure_auth_configured:
        st.error("Authentication not configured")
        return True

    try:
        auth = AzureEntraAuth()
        user = auth.handle_callback(code, state)

        if user:
            set_current_user(user)

            # Sync user to database
            _sync_user_to_db(user)

            # Clear query params and redirect
            st.query_params.clear()
            st.success(f"Welcome, {user.display_name}!")
            st.rerun()
    except Exception as e:
        st.error(f"Authentication failed: {e}")

    return True


def _sync_user_to_db(user: AuthenticatedUser) -> None:
    """Sync authenticated user to database."""
    try:
        from skplay.backend.database import get_db_context
        from skplay.backend.repositories import UserRepository

        with get_db_context() as db:
            repo = UserRepository(db)
            repo.get_or_create_from_azure(
                azure_oid=user.oid,
                email=user.email,
                display_name=user.display_name,
            )
    except Exception:
        # Database might not be configured - that's OK for local development
        pass
