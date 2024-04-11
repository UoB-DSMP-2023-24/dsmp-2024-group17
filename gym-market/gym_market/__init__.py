import logging
from gymnasium.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='Market-v0',
    entry_point='gym_market.envs:Market',

)