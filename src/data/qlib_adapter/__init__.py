"""Qlib data adapter: bin writer, calendar generator, instrument generator.

Data READING is done through Qlib's native API (src.core.data_access).
This module only handles data WRITING (Wind DataFrame → Qlib bin files).
"""
from .converter import WindToQlibConverter
from .calendar import CalendarGenerator
from .instrument import InstrumentGenerator
