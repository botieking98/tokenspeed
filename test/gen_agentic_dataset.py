#!/usr/bin/env python3
"""Generate a synthetic agentic multi-turn dataset for benchmarking."""

import json
import random
import argparse

REPO_CONTEXTS = [
    f"""You are debugging a Python web application. Here is the relevant code:

```python
# file: app/models/user.py
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import hashlib

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(80), unique=True, nullable=False)
    email = Column(String(120), unique=True, nullable=False)
    password_hash = Column(String(128))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    posts = relationship('Post', backref='author', lazy='dynamic')
    comments = relationship('Comment', backref='author', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = hashlib.sha256(password.encode()).hexdigest()

    def check_password(self, password):
        return self.password_hash == hashlib.sha256(password.encode()).hexdigest()

    def to_dict(self):
        return {{
            'id': self.id, 'username': self.username, 'email': self.email,
            'created_at': self.created_at.isoformat(), 'is_active': self.is_active,
            'post_count': self.posts.count(),
        }}

# file: app/routes/auth.py
from flask import Blueprint, request, jsonify, session
from app.models.user import User
from app import db
import jwt, datetime

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter_by(username=data.get('username')).first()
    if user and user.check_password(data.get('password')):
        token = jwt.encode({{'user_id': user.id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
        }}, app.config['SECRET_KEY'])
        return jsonify({{'token': token}})
    return jsonify({{'error': 'Invalid credentials'}}), 401

@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    if User.query.filter_by(username=data.get('username')).first():
        return jsonify({{'error': 'Username already exists'}}), 400
    user = User(username=data['username'], email=data['email'])
    user.set_password(data['password'])
    db.session.add(user)
    db.session.commit()
    return jsonify(user.to_dict()), 201

# file: app/routes/posts.py
from flask import Blueprint, request, jsonify
from app.models.post import Post
from app import db
from app.utils.auth import token_required

posts_bp = Blueprint('posts', __name__)

@posts_bp.route('/posts', methods=['GET'])
def get_posts():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    posts = Post.query.order_by(Post.created_at.desc()).paginate(page=page, per_page=per_page)
    return jsonify({{
        'posts': [p.to_dict() for p in posts.items],
        'total': posts.total, 'pages': posts.pages, 'current_page': posts.page,
    }})

@posts_bp.route('/posts', methods=['POST'])
@token_required
def create_post(current_user):
    data = request.get_json()
    post = Post(title=data['title'], body=data['body'], author_id=current_user.id)
    db.session.add(post); db.session.commit()
    return jsonify(post.to_dict()), 201

@posts_bp.route('/posts/<int:post_id>', methods=['PUT'])
@token_required
def update_post(current_user, post_id):
    post = Post.query.get_or_404(post_id)
    if post.author_id != current_user.id:
        return jsonify({{'error': 'Unauthorized'}}), 403
    data = request.get_json()
    post.title = data.get('title', post.title)
    post.body = data.get('body', post.body)
    db.session.commit()
    return jsonify(post.to_dict())
```

The application uses Flask with SQLAlchemy, JWT authentication. Analyze issues and provide fixes. Variant {i}.
""" for i in range(20)
]

USER_TURN_TEMPLATES = [
    "I'm getting a 500 error on the login endpoint. The traceback shows: TypeError: 'NoneType' object is not subscriptable at line 15 in auth.py. What's wrong?",
    "The /posts endpoint is returning stale data after updates. Users see old post content even after successful PUT requests. How do I fix the caching issue?",
    "I need to add rate limiting to the /register endpoint. We're getting hit by bots creating thousands of accounts. What's the best approach with Flask?",
    "The password hashing is using SHA256 which is insecure. I need to migrate to bcrypt. How do I handle the migration for existing users without downtime?",
    "Our JWT tokens don't have a refresh mechanism. Users are complaining about being logged out every 24 hours. Can you implement a refresh token system?",
    "The paginate() call is causing N+1 queries. Each post.to_dict() triggers a separate query for the author. How do I optimize this with eager loading?",
    "I need to add email verification to the registration flow. New users should receive a verification email and can't post until they verify.",
    "The update_post endpoint has a TOCTOU race condition. Two concurrent requests can both pass the authorization check. How do I add optimistic locking?",
    "We need to add full-text search to posts. The current LIKE query is too slow on our 10M row table. What are the options with PostgreSQL?",
    "I want to add WebSocket support for real-time comments. Users should see new comments appear without refreshing.",
    "The application needs to support file uploads for post attachments. Files should be stored in S3 with presigned URLs.",
    "Our error handling is inconsistent across endpoints. Some return JSON errors, others return HTML. I need a unified error handler.",
]


def generate_conversation(conv_id, num_turns, repo_idx):
    messages = [{"role": "system", "content": REPO_CONTEXTS[repo_idx % len(REPO_CONTEXTS)]}]
    for turn in range(num_turns):
        user_msg = random.choice(USER_TURN_TEMPLATES)
        user_msg += f"\n\n[Context: Turn {turn+1}, session {conv_id}, issue #{random.randint(100,999)}]"
        messages.append({"role": "user", "content": user_msg})
        if turn < num_turns - 1:
            assistant_msg = f"Looking at the code you shared, I can see the issue. " * random.randint(5, 15)
            messages.append({"role": "assistant", "content": assistant_msg})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-conversations", type=int, default=64)
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--output", type=str, default="agentic_dataset.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    conversations = []
    for i in range(args.num_conversations):
        num_turns = random.randint(args.min_turns, args.max_turns)
        repo_idx = i % len(REPO_CONTEXTS)
        messages = generate_conversation(i, num_turns, repo_idx)
        conversations.append({"messages": messages, "num_turns": num_turns})

    sharegpt_entries = []
    for conv in conversations:
        msgs = conv["messages"]
        prompt_parts = []
        for msg in msgs:
            role = msg["role"]
            if role == "system":
                prompt_parts.append(f"[System]\n{msg['content']}\n")
            elif role == "user":
                prompt_parts.append(f"[User]\n{msg['content']}\n")
            elif role == "assistant":
                prompt_parts.append(f"[Assistant]\n{msg['content']}\n")
        sharegpt_entries.append({
            "conversations": [
                {"from": "human", "value": "\n".join(prompt_parts)},
                {"from": "gpt", "value": "I'll analyze the code and provide a solution."},
            ]
        })

    with open(args.output, "w") as f:
        json.dump(sharegpt_entries, f)

    total_turns = sum(c["num_turns"] for c in conversations)
    unique_repos = len(set(i % len(REPO_CONTEXTS) for i in range(args.num_conversations)))
    print(f"Generated {len(conversations)} conversations, {total_turns} total turns")
    print(f"  Unique repo contexts: {unique_repos} (prefix sharing groups)")
    print(f"  Turns range: {args.min_turns}-{args.max_turns}")
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
