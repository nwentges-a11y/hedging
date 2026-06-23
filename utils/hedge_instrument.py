from typing import Optional, Dict
from datetime import date


class HedgeInstrument:
    """
    Class representing a hedge instrument for energy trading and risk management.
    Encapsulates all relevant properties and provides a unique identifier.
    """

    @staticmethod
    def generate_id(product_type: str, load_type: str, start_date: date, end_date: date, region: str, market: str, currency: str, underlying: str) -> str:
        """
        Generate a unique instrument ID string based on key attributes.
        Fields are concatenated for clarity and uniqueness.
        """
        start_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
        end_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
        market_str = market if market else 'NA'  # Market or 'NA' if not specified
        currency_str = currency if currency else 'NA'  # Currency or 'NA' if not specified
        underlying_str = underlying if underlying else 'NA'  # Underlying asset or 'NA'
        return f"{product_type}_{load_type}_{region}_{market_str}_{currency_str}_{underlying_str}_{start_str}_{end_str}"

    def __init__(
        self,
        name: str,
        instrument_type: str,
        product_type: str,
        load_type: str,
        start_date: date,
        end_date: date,
        price: Optional[float] = None,
        volume: float = 0.0,
        region: str = "de",
        market: str = "EEX",
        currency: str = "EUR",
        underlying: str = "power",
        additional_data: Optional[Dict] = None,
        instrument_id: Optional[str] = None
    ):
        """
        Initialize a HedgeInstrument instance.
        All fields are required except additional_data and instrument_id (auto-generated if not provided).
        """
        # Unique identifier for the instrument
        self.instrument_id = instrument_id or self.generate_id(product_type, load_type, start_date, end_date, region, market, currency, underlying)
        # Human-readable name
        self.name = name
        # Type of instrument (e.g., 'future', 'option')
        self.instrument_type = instrument_type
        # Product type (e.g., 'week', 'month', 'year')
        self.product_type = product_type
        # Load type (e.g., 'base', 'peak')
        self.load_type = load_type
        # Start and end dates of the contract/coverage
        self.start_date = start_date
        self.end_date = end_date
        # Price and volume of the contract
        self.price = price
        self.volume = volume
        # Region/country (e.g., 'de', 'fr')
        self.region = region
        # Market or exchange (e.g., 'EEX')
        self.market = market
        # Currency (e.g., 'EUR', 'USD')
        self.currency = currency
        # Underlying asset or index
        self.underlying = underlying
        # Any additional metadata as a dictionary
        self.additional_data = additional_data or {}

    def __repr__(self):
        """
        String representation for easy debugging and logging.
        """
        return (
            f"<HedgeInstrument {self.instrument_id}: {self.name}, {self.instrument_type}, {self.product_type}, {self.load_type}, {self.region}, {self.market}, {self.currency}, {self.underlying}, "
            f"{self.start_date} to {self.end_date}, {self.volume} @ {self.price}>"
        )
