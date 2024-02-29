# -*- coding: utf-8 -*-

from datetime import datetime
import ephem
from typing import List

TYPE_ASTRONOMICAL = "astronomical"

STATE_SPRING = 0
STATE_SUMMER = 1
STATE_FALL = 2
STATE_WINTER = 3

NORTHERN = "north"
SOUTHERN = "south"

HEMISPHERE_SEASON_SWAP = {
    STATE_SPRING: STATE_FALL,
    STATE_SUMMER: STATE_WINTER,
    STATE_FALL: STATE_SPRING,
    STATE_WINTER: STATE_SUMMER
}

def season(date: datetime, hemisphere: str = "north", labels: List[str] = ["Spring", "Summer", "Fall", "Winter"], tracking_type: str = "astronomical") -> str:
    """
    Return the season of the given date depending on the latitude and location.

    Args:
        date: datetime object
        hemisphere: "north" or "south", default is "north"
        labels: array of season names, default is ["Spring", "Summer", "Fall", "Winter"]
        tracking_type: Type of season definition. Options are "meteorological" or "astronomical". Default is "astronomical"
        
    Returns:
        A string with season name
    """
    if tracking_type == TYPE_ASTRONOMICAL:
        spring_start = ephem.next_equinox(str(date.year)).datetime()
        summer_start = ephem.next_solstice(str(date.year)).datetime()
        autumn_start = ephem.next_equinox(spring_start).datetime()
        winter_start = ephem.next_solstice(summer_start).datetime()
    else:
        spring_start = datetime(date.year, 3, 1)
        summer_start = datetime(date.year, 6, 1)
        autumn_start = datetime(date.year, 9, 1)
        winter_start = datetime(date.year, 12, 1)
    
    if spring_start <= date < summer_start:
        season = STATE_SPRING
    elif summer_start <= date < autumn_start:
        season = STATE_SUMMER
    elif autumn_start <= date < winter_start:
        season = STATE_FALL
    elif winter_start <= date or spring_start > date:
        season = STATE_WINTER
    
    # Swap the season if on southern hemisphere
    if hemisphere == SOUTHERN:
        return labels[HEMISPHERE_SEASON_SWAP.get(season, season)]
    return labels[season]
