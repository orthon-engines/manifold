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
# Auth Modal Rendering (using session state for modal control)
# -----------------------------------------------------------------------------

def show_login_modal():
    """Login modal using expander (compatible with all Streamlit versions)."""
    st.session_state.show_login_form = True


def show_signup_modal():
    """Signup modal."""
    st.session_state.show_signup_form = True


def show_upload_modal():
    """Upload modal."""
    st.session_state.show_upload_form = True


def render_login_form():
    """Render login form in sidebar or main area."""
    if not st.session_state.get('show_login_form'):
        return

    with st.sidebar.expander("🔑 Login", expanded=True):
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login", type="primary", key="do_login"):
                if email and password:
                    username = email.split('@')[0]
                    tier = 'academic' if any(email.endswith(d) for d in ['.edu', '.ac.uk']) else 'trial'
                    user = User(username=username, tier=tier, email=email)
                    set_user(user)
                    st.session_state.show_login_form = False
                    st.rerun()
                else:
                    st.error("Enter email/password")
        with col2:
            if st.button("Cancel", key="cancel_login"):
                st.session_state.show_login_form = False
                st.rerun()

        if st.button("Sign Up Instead", key="switch_to_signup"):
            st.session_state.show_login_form = False
            st.session_state.show_signup_form = True
            st.rerun()


def render_signup_form():
    """Render signup form."""
    if not st.session_state.get('show_signup_form'):
        return

    with st.sidebar.expander("📝 Sign Up", expanded=True):
        email = st.text_input("Email", key="signup_email")

        tier_option = st.radio(
            "Type",
            ["Academic (.edu)", "Commercial (soon)"],
            key="signup_tier",
            horizontal=True
        )

        is_academic = "Academic" in tier_option

        if is_academic:
            cite_agree = st.checkbox("I will cite ORTHON", key="cite_agree")
            ramen = st.selectbox("Ramen?", RAMEN_OPTIONS, key="signup_ramen")

            if st.button("Create Account", type="primary", key="do_signup"):
                if not email:
                    st.error("Email required")
                elif not any(email.endswith(d) for d in ['.edu', '.ac.uk', '.edu.au']):
                    st.error("Academic email required")
                elif not cite_agree:
                    st.error("Citation required")
                else:
                    user = User(
                        username=email.split('@')[0],
                        tier="academic",
                        email=email,
                        citation_agreed=True,
                        ramen_preference=ramen,
                    )
                    set_user(user)
                    st.session_state.show_signup_form = False
                    st.rerun()
        else:
            st.info("Commercial coming soon")

        if st.button("Cancel", key="cancel_signup"):
            st.session_state.show_signup_form = False
            st.rerun()


def render_upload_form():
    """Render upload form."""
    if not st.session_state.get('show_upload_form'):
        return

    with st.sidebar.expander("📤 Upload", expanded=True):
        uploaded_file = st.file_uploader(
            "CSV, Parquet, or Excel",
            type=['csv', 'parquet', 'xlsx'],
            key="data_upload"
        )

        user = get_current_user()
        trial_used = st.session_state.get('trial_uploads', 0)

        if user is None or user.tier == 'trial':
            remaining = 3 - trial_used
            if remaining > 0:
                st.caption(f"Trial: {remaining} uploads left")
            else:
                st.warning("Trial limit reached")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload", type="primary", key="do_upload"):
                if uploaded_file:
                    st.session_state.trial_uploads = trial_used + 1
                    if user:
                        user.record_upload()
                    st.session_state.show_upload_form = False
                    st.success("Uploaded!")
                    st.rerun()
                else:
                    st.error("Select a file")
        with col2:
            if st.button("Cancel", key="cancel_upload"):
                st.session_state.show_upload_form = False
                st.rerun()


# -----------------------------------------------------------------------------
# Sidebar Auth Section
# -----------------------------------------------------------------------------

def render_auth_sidebar():
    """Render upload + auth buttons in sidebar."""
    st.sidebar.markdown("---")

    user = get_current_user()

    # Upload button
    if st.sidebar.button("📤 Upload Data", use_container_width=True, key="sidebar_upload"):
        st.session_state.show_upload_form = True
        st.rerun()

    # Auth button / user info
    if user:
        # Logged in - show user info
        tier_name = TIER_CONFIG[user.tier]['name']
        st.sidebar.markdown(f"**👤 {user.username}**")
        st.sidebar.caption(tier_name)

        if st.sidebar.button("Logout", use_container_width=True, key="logout_button"):
            logout()
            st.rerun()
    else:
        # Not logged in - show login button
        if st.sidebar.button("🔑 Login / Sign Up", use_container_width=True, key="sidebar_auth"):
            st.session_state.show_login_form = True
            st.rerun()

    # Render forms if active
    render_login_form()
    render_signup_form()
    render_upload_form()


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

    # Initialize form states if not present
    if 'show_login_form' not in st.session_state:
        st.session_state.show_login_form = False
    if 'show_signup_form' not in st.session_state:
        st.session_state.show_signup_form = False
    if 'show_upload_form' not in st.session_state:
        st.session_state.show_upload_form = False

    return True  # Always allow access - no blocking
