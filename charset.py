"""Shared constants for captcha training."""

CHARSET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_INDEX = 0

CHAR_TO_INDEX = {char: index + 1 for index, char in enumerate(CHARSET)}
INDEX_TO_CHAR = {index + 1: char for index, char in enumerate(CHARSET)}

NUM_CLASSES = len(CHARSET) + 1  # blank + charset
IMG_HEIGHT = 64
IMG_WIDTH = 160
