from __future__ import annotations

import logging

from app.core.config import settings
from app.core.database import SessionLocal
from app.core.security import hash_password
from app.models.cpse import Cpse
from app.models.user import User

logger = logging.getLogger("mira.seed")

DEFAULT_CPSES = [
    {"name": "Indian Oil Corporation Limited", "short_code": "IOCL"},
    {"name": "Bharat Petroleum Corporation Limited", "short_code": "BPCL"},
    {"name": "Chennai Petroleum Corporation Limited", "short_code": "CPCL"},
    {"name": "Steel Authority of India Limited", "short_code": "SAIL"},
    {"name": "NTPC Limited", "short_code": "NTPC"},
]

DEFAULT_USERS = [
    {
        "email": settings.seed_admin_email,
        "password": settings.seed_admin_password,
        "full_name": "System Administrator",
        "role": "admin",
        "cpse_short_code": None,
    },
    {
        "email": "steward@mira.gov.in",
        "password": "Steward@123",
        "full_name": "Data Steward",
        "role": "data_steward",
        "cpse_short_code": "CPCL",
    },
    {
        "email": "reviewer@mira.gov.in",
        "password": "Reviewer@123",
        "full_name": "Material Reviewer",
        "role": "reviewer",
        "cpse_short_code": "IOCL",
    },
    {
        "email": "auditor@mira.gov.in",
        "password": "Auditor@123",
        "full_name": "Compliance Auditor",
        "role": "auditor",
        "cpse_short_code": None,
    },
]


def seed_default_users_and_cpses() -> None:
    """Idempotently seed default CPSEs and baseline demonstration users."""
    db = SessionLocal()
    try:
        cpse_map: dict[str, int] = {}
        for cpse_data in DEFAULT_CPSES:
            existing = db.query(Cpse).filter(Cpse.short_code == cpse_data["short_code"]).first()
            if existing is None:
                cpse_obj = Cpse(name=cpse_data["name"], short_code=cpse_data["short_code"])
                db.add(cpse_obj)
                db.flush()
                cpse_map[cpse_data["short_code"]] = cpse_obj.id
                logger.info("Seeded CPSE: %s", cpse_data["short_code"])
            else:
                cpse_map[cpse_data["short_code"]] = existing.id

        users_to_seed = DEFAULT_USERS if settings.seed_demo_users else DEFAULT_USERS[:1]
        for user_data in users_to_seed:
            existing_user = db.query(User).filter(User.email == user_data["email"].lower()).first()
            cpse_id = cpse_map.get(user_data["cpse_short_code"]) if user_data["cpse_short_code"] else None
            if existing_user is None:
                user_obj = User(
                    email=user_data["email"].lower(),
                    password_hash=hash_password(user_data["password"]),
                    full_name=user_data["full_name"],
                    role=user_data["role"],
                    cpse_id=cpse_id,
                    is_active=True,
                )
                db.add(user_obj)
                logger.info("Seeded default user: %s (%s)", user_data["email"], user_data["role"])
            else:
                existing_user.full_name = user_data["full_name"]
                if cpse_id:
                    existing_user.cpse_id = cpse_id
                db.flush()

        db.commit()
    except Exception as exc:
        db.rollback()
        logger.warning("Default seeding encountered exception: %s", exc)
    finally:
        db.close()
