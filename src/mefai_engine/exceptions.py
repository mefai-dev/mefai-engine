"""Custom exception hierarchy for MEFAI Engine."""


class MefaiError(Exception):
    """Base exception for all MEFAI Engine errors."""


# Exchange errors
class ExchangeError(MefaiError):
    """Base exchange error."""


class ExchangeConnectionError(ExchangeError):
    """Failed to connect to exchange."""


class ExchangeAuthError(ExchangeError):
    """Authentication failed."""


class ExchangeRateLimitError(ExchangeError):
    """Rate limit exceeded."""


class OrderError(ExchangeError):
    """Order placement/management failed."""


class InsufficientBalanceError(OrderError):
    """Not enough balance for order."""


# Data errors
class DataError(MefaiError):
    """Base data error."""


class DataFetchError(DataError):
    """Failed to fetch data from source."""


class DataValidationError(DataError):
    """Data failed validation checks."""


# Model errors
class ModelError(MefaiError):
    """Base model error."""


class ModelNotTrainedError(ModelError):
    """Model has not been trained yet."""


class ModelLoadError(ModelError):
    """Failed to load model from disk."""


class PredictionError(ModelError):
    """Model prediction failed."""


# Strategy errors
class StrategyError(MefaiError):
    """Base strategy error."""


class InsufficientDataError(StrategyError):
    """Not enough data to generate signal."""


# Risk errors
class RiskError(MefaiError):
    """Base risk error."""


class RiskLimitExceeded(RiskError):
    """Risk limit would be exceeded."""


class CircuitBreakerTripped(RiskError):
    """Circuit breaker is in OPEN state."""


# Execution errors
class ExecutionError(MefaiError):
    """Base execution error."""


class SlippageExceeded(ExecutionError):
    """Actual slippage exceeded tolerance."""


class ReconciliationError(ExecutionError):
    """Position reconciliation mismatch."""


# Config errors
class ConfigError(MefaiError):
    """Configuration error."""
