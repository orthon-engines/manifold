"""
ORTHON Authentication & Tier System

Modal-based login/signup flow with upload tracking.
"""

import streamlit as st
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib


# -----------------------------------------------------------------------------
# User Model
# -----------------------------------------------------------------------------

@dataclass
class User:
    username: str
    tier: str  # 'visitor', 'trial', 'academic', 'commercial'
    email: str = ""
    institution: str = ""
    uploads_used: int = 0
    citation_agreed: bool = False
    ramen_preference: str = None
    created_at: datetime = field(default_factory=datetime.now)

    def can_view_demos(self) -> bool:
        return True  # Everyone can view

    def can_upload(self) -> bool:
        if self.tier == 'visitor':
            return False
        if self.tier == 'trial':
            return self.uploads_used < 3
        return True  # academic, commercial

    def uploads_remaining(self) -> Optional[int]:
        if self.tier == 'visitor':
            return 0
        if self.tier == 'trial':
            return max(0, 3 - self.uploads_used)
        return None  # Unlimited

    def record_upload(self):
        self.uploads_used += 1

    def to_dict(self) -> dict:
        return {
            'username': self.username,
            'tier': self.tier,
            'email': self.email,
            'institution': self.institution,
            'uploads_used': self.uploads_used,
            'citation_agreed': self.citation_agreed,
            'ramen_preference': self.ramen_preference,
            'created_at': self.created_at.isoformat(),
        }


# -----------------------------------------------------------------------------
# Tier Configuration
# -----------------------------------------------------------------------------

TIER_CONFIG = {
    'visitor': {
        'name': 'Visitor',
        'can_view_demos': True,
        'can_upload': False,
        'upload_limit': 0,
        'description': 'View demos only',
    },
    'trial': {
        'name': 'Trial',
        'can_view_demos': True,
        'can_upload': True,
        'upload_limit': 3,
        'description': '3 free uploads',
    },
    'academic': {
        'name': 'Academic',
        'can_view_demos': True,
        'can_upload': True,
        'upload_limit': float('inf'),
        'description': 'Unlimited (citation + ramen required)',
        'requirements': ['citation', 'ramen_preference'],
    },
    'commercial': {
        'name': 'Commercial',
        'can_view_demos': True,
        'can_upload': True,
        'upload_limit': float('inf'),
        'description': 'Unlimited (paid)',
        'requirements': ['payment'],
    },
}

RAMEN_OPTIONS = [
    "Tonkotsu (rich pork)",
    "Shoyu (soy sauce)",
    "Miso (fermented soybean)",
    "Shio (salt)",
    "Tantanmen (spicy sesame)",
    "Shin Ramyun (Korean fire)",
    "Maruchan (desperate times)",
    "Indomie Mi Goreng (acceptable)",
]

CITATION_TEXT = """Author, J. (2026). ORTHON: A Domain-Agnostic Framework for
Signal Typology, Structural Geometry, Dynamical Systems, and Causal Mechanics.
[Software]. https://github.com/yourrepo/orthon"""


# -----------------------------------------------------------------------------
# Session State Management
# -----------------------------------------------------------------------------

def init_session_state():
    """Initialize session state for auth."""
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'trial_uploads' not in st.session_state:
        st.session_state.trial_uploads = 0


def get_current_user() -> Optional[User]:
    """Get current user from session state."""
    return st.session_state.get('user')


def set_user(user: User):
    """Set current user in session state."""
    st.session_state.user = user


def logout():
    """Clear user session."""
    st.session_state.user = None


def is_logged_in() -> bool:
    """Check if user is logged in."""
    return st.session_state.get('user') is not None


# -----------------------------------------------------------------------------
# Auth Modal Rendering
# -----------------------------------------------------------------------------

@st.dialog("Login")
def show_login_modal():
    """Login modal dialog."""
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")

    st.markdown("[Forgot Password?](#)")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login", type="primary", use_container_width=True):
            if email and password:
                # Simulate login (in production, validate against database)
                username = email.split('@')[0]
                tier = 'academic' if any(email.endswith(d) for d in ['.edu', '.ac.uk']) else 'trial'
                user = User(username=username, tier=tier, email=email)
                set_user(user)
                st.rerun()
            else:
                st.error("Please enter email and password")

    st.markdown("---")
    st.markdown("Don't have an account?")
    if st.button("Sign Up Free", use_container_width=True):
        st.session_state.show_signup = True
        st.rerun()


@st.dialog("Sign Up")
def show_signup_modal():
    """Signup modal dialog."""
    email = st.text_input("Email", key="signup_email")

    tier_option = st.radio(
        "Account Type",
        ["Academic (.edu, .ac.uk, etc.)", "Commercial (coming soon)"],
        key="signup_tier"
    )

    is_academic = "Academic" in tier_option

    if is_academic:
        st.markdown("---")
        st.markdown("**Academic Requirements:**")

        cite_agree = st.checkbox(
            "I will cite ORTHON in publications using this tool",
            key="cite_agree"
        )

        st.markdown("---")
        ramen = st.selectbox(
            "Ramen preference",
            RAMEN_OPTIONS,
            key="signup_ramen"
        )
        st.caption("*(critical research data)*")

        if st.button("Create Account", type="primary", use_container_width=True):
            errors = []
            if not email:
                errors.append("Email required")
            elif not any(email.endswith(d) for d in ['.edu', '.ac.uk', '.edu.au', '.ac.jp', '.edu.cn']):
                errors.append("Academic email required (.edu, .ac.uk, etc.)")
            if not cite_agree:
                errors.append("Citation agreement required")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                username = email.split('@')[0]
                user = User(
                    username=username,
                    tier="academic",
                    email=email,
                    citation_agreed=True,
                    ramen_preference=ramen,
                )
                set_user(user)
                st.success(f"Welcome! Your ramen preference ({ramen}) has been noted.")
                st.rerun()

    else:
        st.info("Commercial licensing coming soon. Contact us for enterprise inquiries.")
        waitlist_email = st.text_input("Email for waitlist", key="waitlist_email")
        if st.button("Join Waitlist", use_container_width=True):
            if waitlist_email:
                st.success("Added to waitlist!")
            else:
                st.error("Email required")


@st.dialog("Upload Data")
def show_upload_modal():
    """Upload data modal dialog."""
    uploaded_file = st.file_uploader(
        "Drag & drop or browse",
        type=['csv', 'parquet', 'xlsx'],
        key="data_upload"
    )

    format_option = st.radio(
        "Format",
        ["Wide (columns = signals)", "Long (signal_id, timestamp, value)"],
        key="upload_format"
    )

    # Show trial/upload status
    user = get_current_user()
    if user is None or user.tier == 'trial':
        trial_used = st.session_state.get('trial_uploads', 0)
        remaining = 3 - trial_used

        if remaining > 0:
            st.info(f"Trial Mode: {remaining} uploads remaining")
        else:
            st.warning("Trial limit reached!")

        if st.button("Create free account for unlimited", use_container_width=True):
            st.session_state.show_signup = True
            st.rerun()

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Cancel", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Upload & Analyze", type="primary", use_container_width=True):
            if uploaded_file:
                # Process upload
                st.session_state.trial_uploads = st.session_state.get('trial_uploads', 0) + 1

                if user:
                    user.record_upload()

                st.success("File uploaded successfully!")
                st.rerun()
            else:
                st.error("Please select a file")


# -----------------------------------------------------------------------------
# Sidebar Auth Section
# -----------------------------------------------------------------------------

def render_auth_sidebar():
    """Render upload + auth buttons in sidebar."""
    st.sidebar.markdown("---")

    # Upload button
    if st.sidebar.button("📤 Upload Data", use_container_width=True, key="sidebar_upload"):
        show_upload_modal()

    # Auth button / user info
    user = get_current_user()

    if user:
        # Logged in - show user info
        tier_name = TIER_CONFIG[user.tier]['name']

        if st.sidebar.button(f"👤 {user.username}\n{tier_name}", use_container_width=True, key="user_button"):
            # Could show user settings modal
            pass

        if st.sidebar.button("Logout", use_container_width=True, key="logout_button"):
            logout()
            st.rerun()
    else:
        # Not logged in - show login button
        if st.sidebar.button("🔑 Login / Sign Up", use_container_width=True, key="sidebar_auth"):
            show_login_modal()


def render_user_badge():
    """Show current user status in sidebar (compact version)."""
    user = get_current_user()

    if user is None:
        return

    tier_info = TIER_CONFIG[user.tier]

    if user.tier == 'trial':
        remaining = user.uploads_remaining()
        st.sidebar.caption(f"Trial ({remaining} uploads left)")
    elif user.tier == 'academic' and user.ramen_preference:
        st.sidebar.caption(f"🍜 {user.ramen_preference}")
    else:
        st.sidebar.caption(tier_info['name'])


# -----------------------------------------------------------------------------
# Permission Checks
# -----------------------------------------------------------------------------

def check_upload_permission() -> bool:
    """Check if current user can upload. Shows appropriate message if not."""
    user = get_current_user()

    # Trial users (no account) can still upload
    if user is None:
        trial_used = st.session_state.get('trial_uploads', 0)
        if trial_used >= 3:
            st.warning("Trial limit reached. Create a free account for unlimited uploads.")
            return False
        return True

    if not user.can_upload():
        if user.tier == 'visitor':
            st.warning("Visitors can view demos only. Start a trial to upload your own data.")
        elif user.tier == 'trial' and user.uploads_remaining() == 0:
            st.warning("Trial limit reached. Upgrade to academic access for unlimited uploads.")
        return False

    return True


def record_upload():
    """Record an upload for the current user."""
    user = get_current_user()
    if user:
        user.record_upload()
    else:
        st.session_state.trial_uploads = st.session_state.get('trial_uploads', 0) + 1


# -----------------------------------------------------------------------------
# Main Auth Flow
# -----------------------------------------------------------------------------

def auth_flow() -> bool:
    """
    Main auth flow. Call at top of app.py.

    Unlike page-based flow, this just initializes state.
    Users can explore without logging in.

    Returns True always (no blocking auth pages).
    """
    init_session_state()

    # Check for modal triggers
    if st.session_state.get('show_signup'):
        st.session_state.show_signup = False
        show_signup_modal()

    if st.session_state.get('show_login'):
        st.session_state.show_login = False
        show_login_modal()

    if st.session_state.get('show_upload'):
        st.session_state.show_upload = False
        show_upload_modal()

    return True  # Always allow access - no blocking
