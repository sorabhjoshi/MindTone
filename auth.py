"""Signup/login, with basic security hardening:
- passwords bcrypt-hashed (never stored in plain text)
- email format validated, duplicates rejected
- password requires a mix of letters and numbers, not just length
- accounts lock out for a few minutes after repeated failed logins
  (mitigates naive brute-force password guessing)
"""
import datetime
import re

import bcrypt

from db import (User, create_user, get_session, get_user_by_email, get_user_by_username,
                 is_locked_out, record_failed_login, reset_failed_logins, update_password)

USERNAME_RE = re.compile(r'^[a-zA-Z0-9_]{3,32}$')
# Deliberately a pragmatic check (not the full RFC 5322 grammar, which is
# needlessly complex) — this catches the common real mistakes: missing @,
# missing domain, no TLD, stray spaces.
EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
PASSWORD_HAS_LETTER = re.compile(r'[A-Za-z]')
PASSWORD_HAS_DIGIT = re.compile(r'\d')


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
    except ValueError:
        return False


def validate_password_strength(password: str):
    if len(password) < 8:
        return 'Password must be at least 8 characters.'
    if not PASSWORD_HAS_LETTER.search(password):
        return 'Password must include at least one letter.'
    if not PASSWORD_HAS_DIGIT.search(password):
        return 'Password must include at least one number.'
    return None


def signup(username: str, email: str, password: str):
    """Returns (user_or_None, error_message_or_None)."""
    username = username.strip()
    email = email.strip().lower()

    if not USERNAME_RE.match(username):
        return None, 'Username must be 3-32 characters: letters, numbers, underscore only.'
    if not email:
        return None, 'Email is required.'
    if not EMAIL_RE.match(email):
        return None, 'That doesn\'t look like a valid email address.'
    pw_error = validate_password_strength(password)
    if pw_error:
        return None, pw_error

    session = get_session()
    try:
        if get_user_by_username(session, username) is not None:
            return None, 'That username is already taken.'
        if get_user_by_email(session, email) is not None:
            return None, 'An account with that email already exists.'
        user = create_user(session, username, email, hash_password(password))
        return user, None
    finally:
        session.close()


def login(username: str, password: str):
    """Returns (user_or_None, error_message_or_None)."""
    session = get_session()
    try:
        user = get_user_by_username(session, username.strip())
        if user is None:
            return None, 'Incorrect username or password.'
        if is_locked_out(user):
            minutes_left = max(1, int((user.locked_until - datetime.datetime.utcnow())
                                       .total_seconds() // 60) + 1)
            return None, (f'Too many failed attempts. Try again in about '
                           f'{minutes_left} minute(s).')
        if not verify_password(password, user.password_hash):
            record_failed_login(session, user)
            return None, 'Incorrect username or password.'
        reset_failed_logins(session, user)
        return user, None
    finally:
        session.close()


def change_password(user_id: int, current_password: str, new_password: str):
    """Returns (success: bool, error_message_or_None)."""
    pw_error = validate_password_strength(new_password)
    if pw_error:
        return False, pw_error
    session = get_session()
    try:
        user = session.get(User, user_id)
        if user is None or not verify_password(current_password, user.password_hash):
            return False, 'Current password is incorrect.'
        update_password(session, user, hash_password(new_password))
        return True, None
    finally:
        session.close()
