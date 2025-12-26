"""Dataset registry with toy datasets categorized by domain.

Provides a unified interface for loading toy datasets and sklearn built-in datasets.
Each dataset includes metadata: task type, feature info, description, and domain tags.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn import datasets as sklearn_datasets

TaskType = Literal["classification", "regression", "clustering", "outlier_detection"]
Domain = Literal["general", "power", "retail", "finance", "healthcare"]


@dataclass
class FeatureInfo:
    """Information about a single feature."""

    name: str
    dtype: Literal["numeric", "categorical", "binary"]
    description: str = ""
    categories: list[str] | None = None


@dataclass
class DatasetCard:
    """Metadata card for a dataset."""

    name: str
    description: str
    task_type: TaskType
    domain: Domain
    n_samples: int
    n_features: int
    target_name: str | None
    features: list[FeatureInfo]
    source: str = "synthetic"
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    tags: list[str] = field(default_factory=list)


@dataclass
class DatasetResult:
    """Result from loading a dataset."""

    X: pd.DataFrame
    y: pd.Series | None
    card: DatasetCard


class DatasetRegistry:
    """Registry of available toy datasets."""

    _datasets: dict[str, Callable[[], DatasetResult]] = {}
    _cards: dict[str, DatasetCard] = {}

    @classmethod
    def register(cls, name: str, loader: Callable[[], DatasetResult], card: DatasetCard) -> None:
        """Register a dataset with its loader and metadata."""
        cls._datasets[name] = loader
        cls._cards[name] = card

    @classmethod
    def get(cls, name: str) -> DatasetResult:
        """Load and return a dataset by name."""
        if name not in cls._datasets:
            raise ValueError(f"Dataset '{name}' not found. Available: {list(cls._datasets.keys())}")
        return cls._datasets[name]()

    @classmethod
    def list_all(cls) -> list[str]:
        """List all available dataset names."""
        return list(cls._datasets.keys())

    @classmethod
    def list_by_domain(cls, domain: Domain) -> list[str]:
        """List datasets filtered by domain."""
        return [name for name, card in cls._cards.items() if card.domain == domain]

    @classmethod
    def list_by_task(cls, task_type: TaskType) -> list[str]:
        """List datasets filtered by task type."""
        return [name for name, card in cls._cards.items() if card.task_type == task_type]

    @classmethod
    def get_card(cls, name: str) -> DatasetCard:
        """Get the dataset card (metadata) for a dataset."""
        if name not in cls._cards:
            raise ValueError(f"Dataset '{name}' not found.")
        return cls._cards[name]

    @classmethod
    def get_domains(cls) -> list[Domain]:
        """Get all unique domains."""
        return list({card.domain for card in cls._cards.values()})

    @classmethod
    def get_task_types(cls) -> list[TaskType]:
        """Get all unique task types."""
        return list({card.task_type for card in cls._cards.values()})


# =============================================================================
# GENERAL DOMAIN: sklearn built-in datasets
# =============================================================================


def _load_iris() -> DatasetResult:
    """Load iris dataset."""
    data = sklearn_datasets.load_iris(as_frame=True)
    X = data.data
    y = data.target.map(dict(enumerate(data.target_names)))
    y.name = "species"

    features = [
        FeatureInfo(name=col, dtype="numeric", description=f"Iris {col}") for col in X.columns
    ]

    card = DatasetCard(
        name="iris",
        description="Classic iris flower classification. Predict species from sepal/petal measurements.",
        task_type="classification",
        domain="general",
        n_samples=len(X),
        n_features=len(X.columns),
        target_name="species",
        features=features,
        source="sklearn.datasets",
        difficulty="easy",
        tags=["multiclass", "balanced", "small"],
    )

    return DatasetResult(X=X, y=y, card=card)


def _load_wine() -> DatasetResult:
    """Load wine dataset."""
    data = sklearn_datasets.load_wine(as_frame=True)
    X = data.data
    y = data.target.map({i: f"class_{i}" for i in range(3)})
    y.name = "wine_class"

    features = [
        FeatureInfo(name=col, dtype="numeric", description=f"Wine {col}") for col in X.columns
    ]

    card = DatasetCard(
        name="wine",
        description="Wine recognition dataset. Classify wine cultivar from chemical analysis.",
        task_type="classification",
        domain="general",
        n_samples=len(X),
        n_features=len(X.columns),
        target_name="wine_class",
        features=features,
        source="sklearn.datasets",
        difficulty="easy",
        tags=["multiclass", "small"],
    )

    return DatasetResult(X=X, y=y, card=card)


def _load_breast_cancer() -> DatasetResult:
    """Load breast cancer dataset."""
    data = sklearn_datasets.load_breast_cancer(as_frame=True)
    X = data.data
    y = data.target.map({0: "malignant", 1: "benign"})
    y.name = "diagnosis"

    features = [
        FeatureInfo(name=col, dtype="numeric", description=f"Tumor {col}") for col in X.columns
    ]

    card = DatasetCard(
        name="breast_cancer",
        description="Breast cancer diagnosis from tumor characteristics. Binary classification.",
        task_type="classification",
        domain="healthcare",
        n_samples=len(X),
        n_features=len(X.columns),
        target_name="diagnosis",
        features=features,
        source="sklearn.datasets",
        difficulty="easy",
        tags=["binary", "imbalanced"],
    )

    return DatasetResult(X=X, y=y, card=card)


def _load_diabetes() -> DatasetResult:
    """Load diabetes regression dataset."""
    data = sklearn_datasets.load_diabetes(as_frame=True)
    X = data.data
    y = data.target
    y.name = "progression"

    features = [
        FeatureInfo(name=col, dtype="numeric", description=f"Diabetes {col}") for col in X.columns
    ]

    card = DatasetCard(
        name="diabetes",
        description="Predict diabetes progression from baseline measurements.",
        task_type="regression",
        domain="healthcare",
        n_samples=len(X),
        n_features=len(X.columns),
        target_name="progression",
        features=features,
        source="sklearn.datasets",
        difficulty="medium",
        tags=["regression", "continuous"],
    )

    return DatasetResult(X=X, y=y, card=card)


def _load_california_housing() -> DatasetResult:
    """Load California housing dataset."""
    data = sklearn_datasets.fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    y.name = "median_house_value"

    features = [
        FeatureInfo(name=col, dtype="numeric", description=f"Housing {col}") for col in X.columns
    ]

    card = DatasetCard(
        name="california_housing",
        description="Predict California house prices from census data.",
        task_type="regression",
        domain="general",
        n_samples=len(X),
        n_features=len(X.columns),
        target_name="median_house_value",
        features=features,
        source="sklearn.datasets",
        difficulty="medium",
        tags=["regression", "larger"],
    )

    return DatasetResult(X=X, y=y, card=card)


def _load_digits() -> DatasetResult:
    """Load digits dataset for clustering/classification."""
    data = sklearn_datasets.load_digits(as_frame=True)
    X = data.data
    y = data.target.astype(str)
    y.name = "digit"

    features = [
        FeatureInfo(name=col, dtype="numeric", description=f"Pixel {col}") for col in X.columns
    ]

    card = DatasetCard(
        name="digits",
        description="Handwritten digits recognition. Good for clustering and classification.",
        task_type="classification",
        domain="general",
        n_samples=len(X),
        n_features=len(X.columns),
        target_name="digit",
        features=features,
        source="sklearn.datasets",
        difficulty="medium",
        tags=["multiclass", "image", "high-dim"],
    )

    return DatasetResult(X=X, y=y, card=card)


# =============================================================================
# POWER INDUSTRY DOMAIN
# =============================================================================


def _load_power_failure() -> DatasetResult:
    """Synthetic power equipment failure prediction dataset."""
    np.random.seed(42)
    n_samples = 500

    # Generate synthetic sensor data
    temperature = np.random.normal(85, 15, n_samples)
    vibration = np.random.exponential(2, n_samples)
    pressure = np.random.normal(100, 10, n_samples)
    load_pct = np.random.uniform(30, 100, n_samples)
    age_years = np.random.uniform(0, 20, n_samples)
    maintenance_days = np.random.exponential(90, n_samples)

    # Create failure probability based on features
    failure_prob = (
        0.1 * (temperature > 100).astype(float)
        + 0.2 * (vibration > 4).astype(float)
        + 0.15 * (pressure < 80).astype(float)
        + 0.1 * (load_pct > 90).astype(float)
        + 0.1 * (age_years > 15).astype(float)
        + 0.15 * (maintenance_days > 180).astype(float)
        + np.random.uniform(0, 0.2, n_samples)
    )
    failure = (failure_prob > 0.4).astype(int)

    X = pd.DataFrame(
        {
            "temperature_c": temperature,
            "vibration_mm_s": vibration,
            "pressure_psi": pressure,
            "load_percent": load_pct,
            "equipment_age_years": age_years,
            "days_since_maintenance": maintenance_days,
        }
    )
    y = pd.Series(failure, name="failure").map({0: "normal", 1: "failure"})

    features = [
        FeatureInfo("temperature_c", "numeric", "Operating temperature in Celsius"),
        FeatureInfo("vibration_mm_s", "numeric", "Vibration intensity in mm/s"),
        FeatureInfo("pressure_psi", "numeric", "System pressure in PSI"),
        FeatureInfo("load_percent", "numeric", "Load as percentage of capacity"),
        FeatureInfo("equipment_age_years", "numeric", "Age of equipment in years"),
        FeatureInfo("days_since_maintenance", "numeric", "Days since last maintenance"),
    ]

    card = DatasetCard(
        name="power_failure",
        description="Predict power equipment failures from sensor readings and operational data.",
        task_type="classification",
        domain="power",
        n_samples=n_samples,
        n_features=6,
        target_name="failure",
        features=features,
        source="synthetic",
        difficulty="medium",
        tags=["binary", "predictive-maintenance", "imbalanced"],
    )

    return DatasetResult(X=X, y=y, card=card)


def _load_power_efficiency() -> DatasetResult:
    """Synthetic power plant efficiency/heat-rate regression dataset."""
    np.random.seed(43)
    n_samples = 400

    ambient_temp = np.random.uniform(5, 40, n_samples)
    humidity_pct = np.random.uniform(20, 90, n_samples)
    fuel_quality = np.random.uniform(0.8, 1.0, n_samples)
    load_mw = np.random.uniform(100, 500, n_samples)
    cooling_efficiency = np.random.uniform(0.7, 0.95, n_samples)

    # Heat rate depends on these factors (lower is better)
    heat_rate = (
        8000
        + 50 * (ambient_temp - 20)
        + 10 * (humidity_pct - 50)
        - 2000 * (fuel_quality - 0.9)
        - 5 * (load_mw - 300)
        - 1000 * (cooling_efficiency - 0.85)
        + np.random.normal(0, 100, n_samples)
    )

    X = pd.DataFrame(
        {
            "ambient_temp_c": ambient_temp,
            "humidity_percent": humidity_pct,
            "fuel_quality_index": fuel_quality,
            "load_mw": load_mw,
            "cooling_efficiency": cooling_efficiency,
        }
    )
    y = pd.Series(heat_rate, name="heat_rate_btu_kwh")

    features = [
        FeatureInfo("ambient_temp_c", "numeric", "Ambient temperature in Celsius"),
        FeatureInfo("humidity_percent", "numeric", "Relative humidity percentage"),
        FeatureInfo("fuel_quality_index", "numeric", "Fuel quality index (0.8-1.0)"),
        FeatureInfo("load_mw", "numeric", "Power output in megawatts"),
        FeatureInfo("cooling_efficiency", "numeric", "Cooling system efficiency"),
    ]

    card = DatasetCard(
        name="power_efficiency",
        description="Predict power plant heat rate from operating conditions. Lower heat rate = better efficiency.",
        task_type="regression",
        domain="power",
        n_samples=n_samples,
        n_features=5,
        target_name="heat_rate_btu_kwh",
        features=features,
        source="synthetic",
        difficulty="medium",
        tags=["regression", "efficiency", "energy"],
    )

    return DatasetResult(X=X, y=y, card=card)


# =============================================================================
# RETAIL DOMAIN
# =============================================================================


def _load_retail_churn() -> DatasetResult:
    """Synthetic customer churn classification dataset."""
    np.random.seed(44)
    n_samples = 600

    recency_days = np.random.exponential(60, n_samples)
    frequency = np.random.poisson(5, n_samples) + 1
    monetary = np.random.exponential(200, n_samples) + 50
    tenure_months = np.random.uniform(1, 60, n_samples)
    support_tickets = np.random.poisson(1, n_samples)
    satisfaction_score = np.random.uniform(1, 5, n_samples)

    # Churn probability
    churn_prob = (
        0.3 * (recency_days > 90).astype(float)
        + 0.2 * (frequency < 3).astype(float)
        + 0.1 * (monetary < 100).astype(float)
        + 0.15 * (tenure_months < 6).astype(float)
        + 0.1 * (support_tickets > 2).astype(float)
        + 0.15 * (satisfaction_score < 2.5).astype(float)
    )
    churned = (churn_prob + np.random.uniform(0, 0.3, n_samples) > 0.5).astype(int)

    X = pd.DataFrame(
        {
            "recency_days": recency_days,
            "purchase_frequency": frequency,
            "monetary_value": monetary,
            "tenure_months": tenure_months,
            "support_tickets": support_tickets,
            "satisfaction_score": satisfaction_score,
        }
    )
    y = pd.Series(churned, name="churned").map({0: "retained", 1: "churned"})

    features = [
        FeatureInfo("recency_days", "numeric", "Days since last purchase"),
        FeatureInfo("purchase_frequency", "numeric", "Number of purchases in last year"),
        FeatureInfo("monetary_value", "numeric", "Average order value in dollars"),
        FeatureInfo("tenure_months", "numeric", "Customer tenure in months"),
        FeatureInfo("support_tickets", "numeric", "Support tickets in last 6 months"),
        FeatureInfo("satisfaction_score", "numeric", "Customer satisfaction (1-5)"),
    ]

    card = DatasetCard(
        name="retail_churn",
        description="Predict customer churn from RFM metrics and engagement data.",
        task_type="classification",
        domain="retail",
        n_samples=n_samples,
        n_features=6,
        target_name="churned",
        features=features,
        source="synthetic",
        difficulty="medium",
        tags=["binary", "churn", "RFM"],
    )

    return DatasetResult(X=X, y=y, card=card)


def _load_retail_demand() -> DatasetResult:
    """Synthetic product demand regression dataset."""
    np.random.seed(45)
    n_samples = 500

    base_price = np.random.uniform(10, 100, n_samples)
    discount_pct = np.random.uniform(0, 30, n_samples)
    is_weekend = np.random.binomial(1, 0.3, n_samples)
    is_holiday = np.random.binomial(1, 0.1, n_samples)
    competitor_price = base_price * np.random.uniform(0.9, 1.1, n_samples)
    advertising_spend = np.random.exponential(500, n_samples)
    season = np.random.choice(["spring", "summer", "fall", "winter"], n_samples)

    # Demand model
    season_effect = {"spring": 1.0, "summer": 1.2, "fall": 0.9, "winter": 1.1}
    demand = (
        100
        - 0.5 * base_price
        + 2 * discount_pct
        + 20 * is_weekend
        + 30 * is_holiday
        + 0.3 * (competitor_price - base_price)
        + 0.02 * advertising_spend
        + np.array([season_effect[s] for s in season]) * 10
        + np.random.normal(0, 10, n_samples)
    )
    demand = np.maximum(demand, 0)

    X = pd.DataFrame(
        {
            "base_price": base_price,
            "discount_percent": discount_pct,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "competitor_price": competitor_price,
            "advertising_spend": advertising_spend,
            "season": season,
        }
    )
    y = pd.Series(demand, name="units_sold")

    features = [
        FeatureInfo("base_price", "numeric", "Product base price in dollars"),
        FeatureInfo("discount_percent", "numeric", "Discount percentage applied"),
        FeatureInfo("is_weekend", "binary", "Whether sale is on weekend"),
        FeatureInfo("is_holiday", "binary", "Whether sale is on holiday"),
        FeatureInfo("competitor_price", "numeric", "Competitor price for similar product"),
        FeatureInfo("advertising_spend", "numeric", "Weekly advertising spend in dollars"),
        FeatureInfo(
            "season", "categorical", "Season of the year", ["spring", "summer", "fall", "winter"]
        ),
    ]

    card = DatasetCard(
        name="retail_demand",
        description="Predict product demand from pricing, promotions, and seasonality.",
        task_type="regression",
        domain="retail",
        n_samples=n_samples,
        n_features=7,
        target_name="units_sold",
        features=features,
        source="synthetic",
        difficulty="medium",
        tags=["regression", "demand-forecasting", "mixed-types"],
    )

    return DatasetResult(X=X, y=y, card=card)


def _load_retail_segmentation() -> DatasetResult:
    """Synthetic customer segmentation clustering dataset."""
    np.random.seed(46)
    n_samples = 400

    recency = np.random.exponential(45, n_samples)
    frequency = np.random.poisson(8, n_samples) + 1
    monetary = np.random.exponential(300, n_samples) + 20
    avg_basket_size = np.random.uniform(1, 10, n_samples)
    categories_purchased = np.random.poisson(3, n_samples) + 1
    online_ratio = np.random.beta(2, 2, n_samples)

    X = pd.DataFrame(
        {
            "recency_days": recency,
            "frequency": frequency,
            "monetary_total": monetary,
            "avg_basket_size": avg_basket_size,
            "categories_purchased": categories_purchased,
            "online_purchase_ratio": online_ratio,
        }
    )

    features = [
        FeatureInfo("recency_days", "numeric", "Days since last purchase"),
        FeatureInfo("frequency", "numeric", "Total number of purchases"),
        FeatureInfo("monetary_total", "numeric", "Total amount spent"),
        FeatureInfo("avg_basket_size", "numeric", "Average items per purchase"),
        FeatureInfo("categories_purchased", "numeric", "Number of product categories bought"),
        FeatureInfo("online_purchase_ratio", "numeric", "Fraction of purchases made online"),
    ]

    card = DatasetCard(
        name="retail_segmentation",
        description="Customer segmentation using RFM and behavior metrics. Use clustering algorithms.",
        task_type="clustering",
        domain="retail",
        n_samples=n_samples,
        n_features=6,
        target_name=None,
        features=features,
        source="synthetic",
        difficulty="medium",
        tags=["clustering", "RFM", "segmentation"],
    )

    return DatasetResult(X=X, y=None, card=card)


# =============================================================================
# FINANCE DOMAIN
# =============================================================================


def _load_credit_risk() -> DatasetResult:
    """Synthetic credit risk classification dataset."""
    np.random.seed(47)
    n_samples = 700

    income = np.random.lognormal(10.5, 0.5, n_samples)
    debt_to_income = np.random.uniform(0.1, 0.6, n_samples)
    credit_utilization = np.random.beta(2, 5, n_samples)
    num_accounts = np.random.poisson(5, n_samples) + 1
    delinquencies_2yr = np.random.poisson(0.5, n_samples)
    credit_age_years = np.random.uniform(0.5, 25, n_samples)
    employment_length = np.random.exponential(5, n_samples)
    loan_amount = np.random.uniform(5000, 50000, n_samples)

    # Default probability
    default_prob = (
        0.15 * (debt_to_income > 0.4).astype(float)
        + 0.2 * (credit_utilization > 0.7).astype(float)
        + 0.25 * (delinquencies_2yr > 0).astype(float)
        + 0.1 * (credit_age_years < 2).astype(float)
        + 0.1 * (employment_length < 1).astype(float)
        + 0.1 * (loan_amount / income > 0.5).astype(float)
    )
    default = (default_prob + np.random.uniform(0, 0.2, n_samples) > 0.4).astype(int)

    X = pd.DataFrame(
        {
            "annual_income": income,
            "debt_to_income_ratio": debt_to_income,
            "credit_utilization": credit_utilization,
            "num_credit_accounts": num_accounts,
            "delinquencies_2yr": delinquencies_2yr,
            "credit_history_years": credit_age_years,
            "employment_length_years": employment_length,
            "loan_amount": loan_amount,
        }
    )
    y = pd.Series(default, name="default").map({0: "no_default", 1: "default"})

    features = [
        FeatureInfo("annual_income", "numeric", "Annual income in dollars"),
        FeatureInfo("debt_to_income_ratio", "numeric", "Debt to income ratio"),
        FeatureInfo("credit_utilization", "numeric", "Credit utilization ratio (0-1)"),
        FeatureInfo("num_credit_accounts", "numeric", "Number of open credit accounts"),
        FeatureInfo("delinquencies_2yr", "numeric", "Number of delinquencies in last 2 years"),
        FeatureInfo("credit_history_years", "numeric", "Length of credit history in years"),
        FeatureInfo("employment_length_years", "numeric", "Current employment length in years"),
        FeatureInfo("loan_amount", "numeric", "Requested loan amount"),
    ]

    card = DatasetCard(
        name="credit_risk",
        description="Predict loan default risk from financial profile and credit history.",
        task_type="classification",
        domain="finance",
        n_samples=n_samples,
        n_features=8,
        target_name="default",
        features=features,
        source="synthetic",
        difficulty="medium",
        tags=["binary", "credit", "risk"],
    )

    return DatasetResult(X=X, y=y, card=card)


def _load_fraud_detection() -> DatasetResult:
    """Synthetic transaction fraud detection (outlier detection) dataset."""
    np.random.seed(48)
    n_samples = 500

    # Normal transactions
    n_normal = int(n_samples * 0.95)
    n_fraud = n_samples - n_normal

    # Normal transaction patterns
    normal_amount = np.random.lognormal(4, 1, n_normal)
    normal_hour = np.random.choice(range(8, 22), n_normal)
    normal_distance = np.random.exponential(10, n_normal)
    normal_velocity = np.random.uniform(0, 0.5, n_normal)

    # Fraudulent transaction patterns (anomalous)
    fraud_amount = np.random.lognormal(6, 1.5, n_fraud)
    fraud_hour = np.random.choice(range(0, 6), n_fraud)
    fraud_distance = np.random.exponential(100, n_fraud)
    fraud_velocity = np.random.uniform(0.5, 1.0, n_fraud)

    X = pd.DataFrame(
        {
            "amount": np.concatenate([normal_amount, fraud_amount]),
            "hour_of_day": np.concatenate([normal_hour, fraud_hour]),
            "distance_from_home": np.concatenate([normal_distance, fraud_distance]),
            "velocity_kmh": np.concatenate([normal_velocity, fraud_velocity]),
        }
    )
    y = pd.Series([0] * n_normal + [1] * n_fraud, name="is_fraud").map({0: "normal", 1: "fraud"})

    # Shuffle
    idx = np.random.permutation(n_samples)
    X = X.iloc[idx].reset_index(drop=True)
    y = y.iloc[idx].reset_index(drop=True)

    features = [
        FeatureInfo("amount", "numeric", "Transaction amount"),
        FeatureInfo("hour_of_day", "numeric", "Hour of transaction (0-23)"),
        FeatureInfo("distance_from_home", "numeric", "Distance from home location in km"),
        FeatureInfo("velocity_kmh", "numeric", "Transaction velocity score"),
    ]

    card = DatasetCard(
        name="fraud_detection",
        description="Detect fraudulent transactions. Use for outlier/anomaly detection. Labels available for evaluation.",
        task_type="outlier_detection",
        domain="finance",
        n_samples=n_samples,
        n_features=4,
        target_name="is_fraud",
        features=features,
        source="synthetic",
        difficulty="hard",
        tags=["outlier", "anomaly", "imbalanced"],
    )

    return DatasetResult(X=X, y=y, card=card)


# =============================================================================
# Register all datasets
# =============================================================================


def _register_all_datasets():
    """Register all built-in datasets."""
    # General domain (sklearn built-ins)
    DatasetRegistry.register("iris", _load_iris, _load_iris().card)
    DatasetRegistry.register("wine", _load_wine, _load_wine().card)
    DatasetRegistry.register("breast_cancer", _load_breast_cancer, _load_breast_cancer().card)
    DatasetRegistry.register("diabetes", _load_diabetes, _load_diabetes().card)
    DatasetRegistry.register(
        "california_housing", _load_california_housing, _load_california_housing().card
    )
    DatasetRegistry.register("digits", _load_digits, _load_digits().card)

    # Power domain
    DatasetRegistry.register("power_failure", _load_power_failure, _load_power_failure().card)
    DatasetRegistry.register(
        "power_efficiency", _load_power_efficiency, _load_power_efficiency().card
    )

    # Retail domain
    DatasetRegistry.register("retail_churn", _load_retail_churn, _load_retail_churn().card)
    DatasetRegistry.register("retail_demand", _load_retail_demand, _load_retail_demand().card)
    DatasetRegistry.register(
        "retail_segmentation", _load_retail_segmentation, _load_retail_segmentation().card
    )

    # Finance domain
    DatasetRegistry.register("credit_risk", _load_credit_risk, _load_credit_risk().card)
    DatasetRegistry.register("fraud_detection", _load_fraud_detection, _load_fraud_detection().card)


# Initialize registry on import
_register_all_datasets()


# Convenience functions
def get_dataset(name: str) -> DatasetResult:
    """Load a dataset by name."""
    return DatasetRegistry.get(name)


def list_datasets() -> list[str]:
    """List all available datasets."""
    return DatasetRegistry.list_all()


def list_datasets_by_domain(domain: Domain) -> list[str]:
    """List datasets for a specific domain."""
    return DatasetRegistry.list_by_domain(domain)


def list_datasets_by_task(task_type: TaskType) -> list[str]:
    """List datasets for a specific task type."""
    return DatasetRegistry.list_by_task(task_type)
