#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anis Linguistic Radar - الإصدار النهائي المتكامل
Flask-based web application for advanced Arabic text analysis
جميع الحقوق محفوظة للمطور أنيس فيلالي
"""

# ====================== الاستيرادات الأساسية ======================
import os
import sys
import json
import time
import uuid
import hashlib
import logging
from datetime import datetime, timedelta
from functools import wraps
from collections import Counter, OrderedDict
from typing import Dict, List, Tuple, Optional, Any

# Flask及相关扩展
try:
    from flask import Flask, render_template, request, jsonify, url_for, session, redirect, flash
    from flask_sqlalchemy import SQLAlchemy
    from flask_migrate import Migrate
    from flask_caching import Cache
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    from flask_cors import CORS
    from werkzeug.security import generate_password_hash, check_password_hash
    import jwt
except ImportError as e:
    print(f"❌ مكتبة Flask غير مكتملة: {e}")
    print("قم بتثبيت المتطلبات: pip install flask flask-sqlalchemy flask-migrate flask-caching flask-limiter flask-cors pyjwt")
    sys.exit(1)

# مكتبات علمية وتحليلية
try:
    import numpy as np
except ImportError:
    print("❌ numpy غير مثبت. قم بتثبيته عبر: pip install numpy")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')  # وضع غير تفاعلي
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
except ImportError:
    print("❌ matplotlib غير مثبت. قم بتثبيته عبر: pip install matplotlib")
    sys.exit(1)

# مكتبات اللغة العربية المتقدمة (اختياري)
ARABIC_LINGUISTICS_AVAILABLE = False
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    ARABIC_SHAPING_AVAILABLE = True
except ImportError:
    ARABIC_SHAPING_AVAILABLE = False
    arabic_reshaper = None
    get_display = None

try:
    from camel_tools.sentiment import SentimentAnalyzer
    from camel_tools.morphology.database import MorphologyDB
    from camel_tools.morphology.analyzer import Analyzer
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    SentimentAnalyzer = None
    MorphologyDB = None
    Analyzer = None

try:
    from farasa.segmenter import FarasaSegmenter
    from farasa.pos import FarasaPOS
    FARASA_AVAILABLE = True
except ImportError:
    FARASA_AVAILABLE = False
    FarasaSegmenter = None
    FarasaPOS = None

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    WordCloud = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    TfidfVectorizer = None
    cosine_similarity = None

# تخزين سحابي (اختياري)
try:
    import boto3
    from botocore.exceptions import NoCredentialsError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    boto3 = None

# Redis (اختياري)
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# متغيرات البيئة
from dotenv import load_dotenv
load_dotenv()

# ====================== إعدادات التطبيق ======================
class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', os.urandom(24).hex())
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES = int(os.environ.get('JWT_ACCESS_TOKEN_EXPIRES', 3600))  # ساعة

    # قاعدة البيانات
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///anis_radar.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': int(os.environ.get('DB_POOL_SIZE', 10)),
        'pool_recycle': int(os.environ.get('DB_POOL_RECYCLE', 3600)),
    }

    # Redis للتخزين المؤقت ومعدل الطلبات
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
    if REDIS_AVAILABLE and REDIS_URL:
        CACHE_TYPE = 'RedisCache'
        CACHE_REDIS_URL = REDIS_URL
        CACHE_DEFAULT_TIMEOUT = 300
        RATELIMIT_STORAGE_URL = REDIS_URL
    else:
        CACHE_TYPE = 'SimpleCache'
        CACHE_DEFAULT_TIMEOUT = 300
        RATELIMIT_STORAGE_URL = 'memory://'

    RATELIMIT_ENABLED = os.environ.get('RATELIMIT_ENABLED', 'True').lower() == 'true'
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '100 per day, 10 per minute')

    # حدود المحتوى
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))  # 16 MB
    MIN_TEXT_LENGTH = 10
    MAX_TEXT_LENGTH = 20000

    # مسار الخط العربي
    FONT_PATH = os.environ.get('FONT_PATH', 'Amiri-Regular.ttf')

    # تخزين الصور (محلي أو S3)
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'static/images')
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    USE_S3 = os.environ.get('USE_S3', 'False').lower() == 'true'
    if USE_S3 and S3_AVAILABLE:
        S3_BUCKET = os.environ.get('S3_BUCKET')
        S3_KEY = os.environ.get('S3_KEY')
        S3_SECRET = os.environ.get('S3_SECRET')
        S3_REGION = os.environ.get('S3_REGION', 'us-east-1')
        S3_PUBLIC_URL = os.environ.get('S3_PUBLIC_URL', f'https://{S3_BUCKET}.s3.amazonaws.com/')

# ====================== تهيئة التطبيق والامتدادات ======================
app = Flask(__name__)
app.config.from_object(Config)

# قاعدة البيانات
db = SQLAlchemy(app)
migrate = Migrate(app, db)

# التخزين المؤقت
cache = Cache(app)

# الحد من معدل الطلبات
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    storage_uri=app.config['RATELIMIT_STORAGE_URL'],
    enabled=app.config['RATELIMIT_ENABLED']
)

# CORS للواجهة الأمامية (إذا كانت منفصلة)
CORS(app)

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO if not app.debug else logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ====================== نماذج قاعدة البيانات ======================
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    analyses = db.relationship('Analysis', backref='user', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def generate_token(self, expires_in=None):
        if expires_in is None:
            expires_in = app.config['JWT_ACCESS_TOKEN_EXPIRES']
        payload = {
            'user_id': self.id,
            'username': self.username,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in)
        }
        return jwt.encode(payload, app.config['JWT_SECRET_KEY'], algorithm='HS256')

    @staticmethod
    def verify_token(token):
        try:
            payload = jwt.decode(token, app.config['JWT_SECRET_KEY'], algorithms=['HS256'])
            return User.query.get(payload['user_id'])
        except (jwt.ExpiredSignatureError, jwt.InvalidTokenError):
            return None

class Analysis(db.Model):
    __tablename__ = 'analyses'
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    text_preview = db.Column(db.String(200))
    text_hash = db.Column(db.String(64), unique=True, index=True)
    sentiment = db.Column(db.String(20))
    confidence = db.Column(db.Float)
    emotions = db.Column(db.JSON)          # تخزين كـ JSON
    stats = db.Column(db.JSON)              # الإحصائيات الأساسية
    advanced_stats = db.Column(db.JSON)     # إحصائيات متقدمة (مثل TTR, hapax)
    linguistic_features = db.Column(db.JSON)  # الميزات اللغوية المتقدمة
    keywords = db.Column(db.JSON)
    plot_urls = db.Column(db.JSON)          # روابط الصور (إذا خُزنت خارجياً)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)

# ====================== دوال مساعدة للعربية والرسوم ======================
def reshape_arabic(text):
    if not text or not ARABIC_SHAPING_AVAILABLE:
        return text
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception as e:
        logging.error(f"خطأ في reshape: {e}")
        return text

# إعداد الخط
font_prop = None
if os.path.exists(app.config['FONT_PATH']):
    try:
        font_prop = fm.FontProperties(fname=app.config['FONT_PATH'])
        logging.info("✅ تم تحميل الخط العربي")
    except Exception as e:
        logging.warning(f"⚠️ فشل تحميل الخط: {e}")
else:
    logging.warning("⚠️ خط Amiri غير موجود.")

# مجموعات الحروف
VOICED = set("بجتدذرزضظعغقلمنوي")
VOICELESS = set("حثسصشفكهت")
PUNCTUATIONS = set(".,;:!?؟،؛")
RADAR_CATEGORIES = ["الإنتروبيا", "التوازن الصوتي", "الجهر", "الهمس", "طول الكلمة", "ثراء المفردات"]

# ====================== خدمات التحليل المتقدمة ======================
class FeatureExtractor:
    @staticmethod
    def extract(text):
        """استخراج الإحصائيات الأساسية"""
        if not text or len(text.strip()) == 0:
            return [0.0] * 8

        words = text.split()
        total_words = len(words)
        total_chars = len(text)

        # الإنتروبيا
        char_counts = Counter(text)
        if total_chars > 0:
            entropy = -sum((c/total_chars) * np.log2(c/total_chars) for c in char_counts.values() if c > 0)
        else:
            entropy = 0.0

        # التوازن الصوتي
        voiced = sum(1 for c in text if c in VOICED)
        voiceless = sum(1 for c in text if c in VOICELESS)
        total_phonemes = voiced + voiceless
        if total_phonemes:
            voiced_pct = voiced / total_phonemes * 100
            voiceless_pct = voiceless / total_phonemes * 100
            balance = 1 - abs(voiced_pct - voiceless_pct) / 100
        else:
            voiced_pct = voiceless_pct = 50.0
            balance = 1.0

        avg_word = np.mean([len(w) for w in words]) if words else 0.0
        sentences = [s.strip() for s in text.replace('!', '.').replace('؟', '.').replace('،', '.').split('.') if s.strip()]
        avg_sentence = total_words / len(sentences) if sentences else total_words
        unique_words = len(set(words))
        richness = unique_words / total_words if total_words else 0.0
        punct_count = sum(1 for c in text if c in PUNCTUATIONS)
        punct_ratio = punct_count / total_chars if total_chars else 0.0

        return [
            entropy,
            balance,
            voiced_pct,
            voiceless_pct,
            avg_word,
            avg_sentence,
            punct_ratio * 100,
            richness * 100
        ]

    @staticmethod
    def advanced_stylometry(text):
        words = text.split()
        total = len(words)
        unique = len(set(words))
        ttr = unique / total if total else 0.0
        freq = Counter(words)
        hapax = sum(1 for w in freq if freq[w] == 1)
        hapax_ratio = hapax / total if total else 0.0
        content_words = [w for w in words if len(w) > 3]
        lexical_density = len(content_words) / total if total else 0.0
        return ttr, hapax_ratio, lexical_density

class AdvancedLinguisticAnalyzer:
    def __init__(self):
        self.camel_analyzer = None
        self.farasa_segmenter = None
        self.farasa_pos = None

        if CAMEL_AVAILABLE and MorphologyDB and Analyzer:
            try:
                db = MorphologyDB.builtin_db()
                self.camel_analyzer = Analyzer(db)
                logging.info("✅ camel-tools للتحليل الصرفي جاهز")
            except Exception as e:
                logging.error(f"فشل تحميل camel-tools: {e}")

        if FARASA_AVAILABLE and FarasaSegmenter and FarasaPOS:
            try:
                self.farasa_segmenter = FarasaSegmenter(interactive=True)
                self.farasa_pos = FarasaPOS(interactive=True)
                logging.info("✅ Farasa للتحليل النحوي جاهز")
            except Exception as e:
                logging.error(f"فشل تحميل Farasa: {e}")

    def analyze_morphology(self, text: str, max_words: int = 50) -> List[Dict]:
        """تحليل صرفي: جذر، وزن، نوع"""
        if self.camel_analyzer:
            try:
                words = text.split()[:max_words]
                results = []
                for word in words:
                    analyses = self.camel_analyzer.analyze(word)
                    if analyses:
                        first = analyses[0]
                        results.append({
                            'word': word,
                            'root': first.get('root', ''),
                            'pattern': first.get('pattern', ''),
                            'pos': first.get('pos', ''),
                            'gender': first.get('gen', ''),
                            'number': first.get('num', '')
                        })
                    else:
                        results.append({'word': word, 'root': '', 'pattern': '', 'pos': ''})
                return results
            except Exception as e:
                logging.error(f"خطأ في التحليل الصرفي: {e}")
        # محلل بديل بسيط
        return [{'word': w, 'root': w[:3], 'pattern': 'فعل', 'pos': 'اسم'} for w in text.split()[:10]]

    def analyze_syntax(self, text: str) -> Dict:
        """تحليل نحوي: تقطيع الجمل وأجزاء الكلام"""
        if self.farasa_pos and self.farasa_segmenter:
            try:
                sentences = [s.strip() for s in text.replace('؟', '.').replace('!', '.').split('.') if s.strip()]
                syntax_data = []
                for sent in sentences[:5]:  # حد أقصى 5 جمل
                    if sent:
                        tokens = self.farasa_segmenter.segment(sent)
                        pos_tags = self.farasa_pos.tag(sent)
                        syntax_data.append({
                            'sentence': sent,
                            'tokens': tokens,
                            'pos_tags': pos_tags
                        })
                return {'sentences': syntax_data}
            except Exception as e:
                logging.error(f"خطأ في التحليل النحوي: {e}")
        return {'sentences': []}

    def analyze_phonetics(self, text: str) -> Dict:
        """تحليل صوتي إضافي (نسب الحروف حسب المخارج)"""
        # قوائم مبسطة للمخارج
        makharij = {
            'حلقية': set('أهعحغخ'),
            'لهوية': set('قك'),
            'شجرية': set('جشض'),
            'لثوية': set('صسز'),
            'نطعية': set('طدت'),
            'ذلقية': set('لرنبفم')
        }
        total_chars = len([c for c in text if c.strip()])
        if total_chars == 0:
            return {}
        result = {}
        for name, letters in makharij.items():
            count = sum(1 for c in text if c in letters)
            result[name] = round(count / total_chars * 100, 1)
        return result

feature_extractor = FeatureExtractor()
linguistic_analyzer = AdvancedLinguisticAnalyzer()

# محلل المشاعر
class SentimentAnalyzerModule:
    def __init__(self):
        self.advanced = None
        if CAMEL_AVAILABLE and SentimentAnalyzer:
            try:
                self.advanced = SentimentAnalyzer.pretrained()
                logging.info("✅ نموذج المشاعر المتقدم جاهز")
            except Exception as e:
                logging.warning(f"⚠️ فشل تحميل النموذج المتقدم: {e}")

    def analyze(self, text):
        if self.advanced:
            try:
                result = self.advanced.predict([text[:512]])[0]
                emotions = self._extract_emotions(text)
                return result, 0.85, emotions
            except Exception as e:
                logging.error(f"خطأ في التحليل المتقدم: {e}")

        # محلل بسيط
        pos_words = {'حب', 'سعيد', 'فرح', 'جميل', 'رائع', 'ممتاز', 'يبتسم', 'أمل', 'تفاؤل', 'نور',
                     'بهجة', 'سرور', 'لطيف', 'عظيم', 'مبدع', 'ناجح', 'مشرق'}
        neg_words = {'حزن', 'بكاء', 'ألم', 'كئيب', 'مؤلم', 'سيء', 'قبيح', 'ظلام', 'خوف', 'فزع',
                     'صعب', 'عسير', 'مزعج', 'غضب', 'كراهية', 'حقد', 'ضيق', 'هم', 'كارثة'}
        words = text.split()
        pos_count = sum(1 for w in words if w in pos_words)
        neg_count = sum(1 for w in words if w in neg_words)
        total = pos_count + neg_count
        if total == 0:
            return "محايد", 0.5, {}
        pos_ratio = pos_count / total
        neg_ratio = neg_count / total
        if pos_ratio > 0.66:
            sentiment = "إيجابي"
        elif neg_ratio > 0.66:
            sentiment = "سلبي"
        else:
            sentiment = "محايد"
        confidence = max(pos_ratio, neg_ratio)
        emotions = {'positive': pos_count, 'negative': neg_count}
        return sentiment, confidence, emotions

    def _extract_emotions(self, text):
        emotions = {'فرح': 0, 'حزن': 0, 'غضب': 0, 'مفاجأة': 0, 'خوف': 0}
        joy_words = ['سعيد', 'فرح', 'مبسوط', 'يبتسم', 'جميل', 'رائع']
        sad_words = ['حزين', 'بكاء', 'ألم', 'كئيب', 'مؤلم']
        anger_words = ['غاضب', 'غضب', 'كره', 'حقد', 'مزعج']
        surprise_words = ['مفاجأة', 'مذهل', 'عجيب', 'غريب']
        fear_words = ['خائف', 'خوف', 'فزع', 'مرعوب']
        words = text.split()
        for word in words:
            if word in joy_words:
                emotions['فرح'] += 1
            elif word in sad_words:
                emotions['حزن'] += 1
            elif word in anger_words:
                emotions['غضب'] += 1
            elif word in surprise_words:
                emotions['مفاجأة'] += 1
            elif word in fear_words:
                emotions['خوف'] += 1
        return emotions

sentiment_analyzer = SentimentAnalyzerModule()

# ====================== دوال إنشاء الرسوم البيانية (مع دعم S3) ======================
def upload_to_s3(file_path, object_name=None):
    """رفع ملف إلى S3 وإرجاع الرابط العام"""
    if not app.config['USE_S3'] or not S3_AVAILABLE:
        return None
    if object_name is None:
        object_name = os.path.basename(file_path)
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=app.config['S3_KEY'],
            aws_secret_access_key=app.config['S3_SECRET'],
            region_name=app.config['S3_REGION']
        )
        s3_client.upload_file(file_path, app.config['S3_BUCKET'], object_name, ExtraArgs={'ACL': 'public-read'})
        return f"{app.config['S3_PUBLIC_URL']}{object_name}"
    except Exception as e:
        logging.error(f"فشل رفع الملف إلى S3: {e}")
        return None

def create_radar_chart(stats, filename, dark_mode=True):
    try:
        values = [
            stats[0] * 10,
            stats[1] * 100,
            stats[2],
            stats[3],
            stats[4] * 2,
            stats[7]
        ]
        values += values[:1]
        angles = np.linspace(0, 2*np.pi, len(RADAR_CATEGORIES), endpoint=False).tolist()
        angles += angles[:1]

        bg = "#0B0F19" if dark_mode else "#f0f0f0"
        text_color = "#F7FAFC" if dark_mode else "#333333"

        fig = plt.figure(figsize=(6,5), dpi=80, facecolor=bg)
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor("#1a1f2e" if dark_mode else "#e0e0e0")
        ax.plot(angles, values, color="#D4AF37", linewidth=3, marker='o')
        ax.fill(angles, values, color="#D4AF37", alpha=0.3)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])

        categories_reshaped = [reshape_arabic(cat) for cat in RADAR_CATEGORIES]
        if font_prop:
            ax.set_xticklabels(categories_reshaped, fontproperties=font_prop, color=text_color, size=10)
        else:
            ax.set_xticklabels(categories_reshaped, color=text_color, size=10)

        fig.savefig(filename, bbox_inches='tight', facecolor=bg)
        plt.close(fig)
        return True
    except Exception as e:
        logging.error(f"فشل إنشاء الرادار: {e}")
        return False

def create_bar_chart(text, filename, dark_mode=True):
    try:
        words = text.split()
        word_lengths = [len(w) for w in words if w]
        if not word_lengths:
            word_lengths = [0]
        bins = range(1, 12)
        hist, _ = np.histogram(word_lengths, bins=bins)
        labels = [f"{i}-{i+1}" for i in range(1, 11)]

        bg = "#0B0F19" if dark_mode else "#f0f0f0"
        text_color = "#F7FAFC" if dark_mode else "#333333"

        fig = plt.figure(figsize=(6,5), dpi=80, facecolor=bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1f2e" if dark_mode else "#e0e0e0")
        ax.bar(labels, hist, color="#4FD1C5", edgecolor="#D4AF37", linewidth=1.5)

        xlabel = reshape_arabic("طول الكلمة (حروف)")
        ylabel = reshape_arabic("عدد الكلمات")
        title = reshape_arabic("توزيع أطوال الكلمات")

        if font_prop:
            ax.set_xlabel(xlabel, fontproperties=font_prop, color=text_color)
            ax.set_ylabel(ylabel, fontproperties=font_prop, color=text_color)
            ax.set_title(title, fontproperties=font_prop, color="#D4AF37")
        else:
            ax.set_xlabel(xlabel, color=text_color)
            ax.set_ylabel(ylabel, color=text_color)
            ax.set_title(title, color="#D4AF37")

        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color("#D4AF37")
        fig.savefig(filename, bbox_inches='tight', facecolor=bg)
        plt.close(fig)
        return True
    except Exception as e:
        logging.error(f"فشل إنشاء المخطط الشريطي: {e}")
        return False

def create_wordcloud(text, filename, dark_mode=True):
    bg = "#0B0F19" if dark_mode else "#f0f0f0"
    if not WORDCLOUD_AVAILABLE:
        fig = plt.figure(figsize=(6,5), facecolor=bg)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "مكتبة wordcloud غير مثبتة", color='red', ha='center', va='center')
        ax.axis("off")
        fig.savefig(filename, bbox_inches='tight', facecolor=bg)
        plt.close(fig)
        return True

    try:
        reshaped = reshape_arabic(text)
        wc = WordCloud(
            width=500, height=400,
            background_color='#0D1117' if dark_mode else '#f0f0f0',
            font_path=app.config['FONT_PATH'] if os.path.exists(app.config['FONT_PATH']) else None,
            colormap='viridis',
            random_state=42
        ).generate(reshaped)

        fig = plt.figure(figsize=(6,5), dpi=80, facecolor=bg)
        ax = fig.add_subplot(111)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        title = reshape_arabic("سحابة الكلمات")
        if font_prop:
            ax.set_title(title, fontproperties=font_prop, color="#D4AF37", fontsize=14)
        else:
            ax.set_title(title, color="#D4AF37", fontsize=14)
        fig.savefig(filename, bbox_inches='tight', facecolor=bg)
        plt.close(fig)
        return True
    except Exception as e:
        logging.error(f"فشل إنشاء سحابة الكلمات: {e}")
        fig = plt.figure(figsize=(6,5), facecolor=bg)
        ax = fig.add_subplot(111)
        ax.text(0.5,0.5, f"خطأ: {e}", color='red', ha='center', va='center')
        ax.axis("off")
        fig.savefig(filename, bbox_inches='tight', facecolor=bg)
        plt.close(fig)
        return True

# ====================== دوال استخراج الكلمات المفتاحية ======================
def extract_keywords(text, n=5):
    if not SKLEARN_AVAILABLE or TfidfVectorizer is None:
        return ["غير متاح"]
    try:
        vectorizer = TfidfVectorizer(max_features=n)
        sentences = text.split('.') if '.' in text else [text]
        tfidf = vectorizer.fit_transform(sentences)
        return vectorizer.get_feature_names_out().tolist()
    except Exception as e:
        logging.warning(f"فشل استخراج الكلمات المفتاحية: {e}")
        return ["غير متاح"]

# ====================== نظام الكاش (Redis) ======================
class AnalysisCache:
    def __init__(self, cache_obj):
        self.cache = cache_obj

    def get(self, text):
        key = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return self.cache.get(key)

    def set(self, text, value, timeout=300):
        key = hashlib.sha256(text.encode('utf-8')).hexdigest()
        self.cache.set(key, value, timeout=timeout)

analysis_cache = AnalysisCache(cache)

# ====================== مصادقة JWT ======================
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            return jsonify({'error': 'Token missing or invalid'}), 401
        token = token[7:]
        user = User.verify_token(token)
        if not user:
            return jsonify({'error': 'Invalid or expired token'}), 401
        return f(user=user, *args, **kwargs)
    return decorated

# ====================== نقاط النهاية ======================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/compare')
def compare_page():
    return render_template('compare.html')

# واجهة برمجة التطبيقات (API)

@app.route('/api/analyze', methods=['POST'])
@limiter.limit("10 per minute")
def api_analyze():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'لا يوجد نص'}), 400

    text = data['text'].strip()
    if len(text) < app.config['MIN_TEXT_LENGTH']:
        return jsonify({'error': f'النص قصير جداً (الحد الأدنى {app.config["MIN_TEXT_LENGTH"]} حرف)'}), 400
    if len(text) > app.config['MAX_TEXT_LENGTH']:
        text = text[:app.config['MAX_TEXT_LENGTH']]

    # البحث في الكاش
    cached = analysis_cache.get(text)
    if cached:
        return jsonify({'success': True, 'from_cache': True, 'data': cached})

    # تحليل النص
    stats = feature_extractor.extract(text)
    sentiment, confidence, emotions = sentiment_analyzer.analyze(text)
    ttr, hapax, lex = feature_extractor.advanced_stylometry(text)
    keywords = extract_keywords(text)
    morphology = linguistic_analyzer.analyze_morphology(text)
    syntax = linguistic_analyzer.analyze_syntax(text)
    phonetics = linguistic_analyzer.analyze_phonetics(text)

    # إنشاء الصور
    plot_id = str(uuid.uuid4())
    dark_mode = data.get('dark_mode', True)
    local_files = []

    radar_file = os.path.join(app.config['UPLOAD_FOLDER'], f'radar_{plot_id}.png')
    if create_radar_chart(stats, radar_file, dark_mode):
        local_files.append(radar_file)

    bar_file = os.path.join(app.config['UPLOAD_FOLDER'], f'bar_{plot_id}.png')
    if create_bar_chart(text, bar_file, dark_mode):
        local_files.append(bar_file)

    wc_file = os.path.join(app.config['UPLOAD_FOLDER'], f'wc_{plot_id}.png')
    if create_wordcloud(text, wc_file, dark_mode):
        local_files.append(wc_file)

    # رفع الصور إلى S3 إذا كان مفعلاً
    plot_urls = {}
    if app.config['USE_S3'] and S3_AVAILABLE:
        for f in local_files:
            base = os.path.basename(f)
            url = upload_to_s3(f, base)
            if url:
                if 'radar' in base:
                    plot_urls['radar'] = url
                elif 'bar' in base:
                    plot_urls['bar'] = url
                elif 'wc' in base:
                    plot_urls['wordcloud'] = url
    else:
        # استخدام المسار المحلي
        base_url = '/static/images/'
        plot_urls = {
            'radar': base_url + f'radar_{plot_id}.png' if os.path.exists(radar_file) else None,
            'bar': base_url + f'bar_{plot_id}.png' if os.path.exists(bar_file) else None,
            'wordcloud': base_url + f'wc_{plot_id}.png' if os.path.exists(wc_file) else None
        }

    # إعداد البيانات للتخزين في قاعدة البيانات
    analysis_data = {
        'text_preview': text[:200] + '...' if len(text) > 200 else text,
        'text_hash': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'sentiment': sentiment,
        'confidence': confidence,
        'emotions': emotions,
        'stats': {
            'entropy': round(stats[0], 2),
            'balance': round(stats[1]*100, 1),
            'voiced': round(stats[2], 1),
            'voiceless': round(stats[3], 1),
            'avg_word': round(stats[4], 2),
            'avg_sentence': round(stats[5], 2),
            'punct_ratio': round(stats[6], 1),
            'richness': round(stats[7], 1)
        },
        'advanced_stats': {
            'ttr': round(ttr*100, 1),
            'hapax': round(hapax*100, 1),
            'lexical_density': round(lex*100, 1)
        },
        'linguistic_features': {
            'morphology': morphology,
            'syntax': syntax,
            'phonetics': phonetics
        },
        'keywords': keywords,
        'plot_urls': plot_urls
    }

    # حفظ في قاعدة البيانات (غير متزامن - يمكن تحسينه باستخدام Celery)
    try:
        analysis = Analysis(
            id=str(uuid.uuid4()),
            text_preview=analysis_data['text_preview'],
            text_hash=analysis_data['text_hash'],
            sentiment=sentiment,
            confidence=confidence,
            emotions=emotions,
            stats=analysis_data['stats'],
            advanced_stats=analysis_data['advanced_stats'],
            linguistic_features=analysis_data['linguistic_features'],
            keywords=keywords,
            plot_urls=plot_urls
        )
        db.session.add(analysis)
        db.session.commit()
    except Exception as e:
        logging.error(f"فشل حفظ التحليل في قاعدة البيانات: {e}")
        db.session.rollback()

    # تخزين في الكاش
    analysis_cache.set(text, analysis_data, timeout=300)

    return jsonify({'success': True, 'from_cache': False, 'data': analysis_data})

@app.route('/api/compare', methods=['POST'])
@limiter.limit("10 per minute")
def api_compare():
    data = request.get_json()
    text1 = data.get('text1', '').strip()
    text2 = data.get('text2', '').strip()
    if not text1 or not text2:
        return jsonify({'error': 'الرجاء إدخال كلا النصين'}), 400

    if not SKLEARN_AVAILABLE or cosine_similarity is None:
        return jsonify({'error': 'مكتبة sklearn غير متاحة للمقارنة'}), 503

    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([text1, text2])
        sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100

        stats1 = feature_extractor.extract(text1)
        stats2 = feature_extractor.extract(text2)

        result = {
            'similarity': round(sim, 2),
            'stats1': {
                'entropy': round(stats1[0], 2),
                'richness': round(stats1[7], 1)
            },
            'stats2': {
                'entropy': round(stats2[0], 2),
                'richness': round(stats2[7], 1)
            }
        }
        return jsonify(result)
    except Exception as e:
        logging.error(f"خطأ في المقارنة: {e}")
        return jsonify({'error': 'حدث خطأ داخلي'}), 500

@app.route('/api/history', methods=['GET'])
@token_required
def api_history(user):
    """إرجاع آخر تحليلات المستخدم (مع مصادقة)"""
    analyses = user.analyses.order_by(Analysis.created_at.desc()).limit(50).all()
    history_list = []
    for a in analyses:
        history_list.append({
            'id': a.id,
            'text_preview': a.text_preview,
            'sentiment': a.sentiment,
            'stats': a.stats,
            'created_at': a.created_at.isoformat(),
            'plot_urls': a.plot_urls
        })
    return jsonify(history_list)

@app.route('/api/analysis/<analysis_id>', methods=['GET'])
@token_required
def api_get_analysis(user, analysis_id):
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user.id).first()
    if not analysis:
        return jsonify({'error': 'التحليل غير موجود'}), 404
    return jsonify({
        'id': analysis.id,
        'text_preview': analysis.text_preview,
        'sentiment': analysis.sentiment,
        'stats': analysis.stats,
        'advanced_stats': analysis.advanced_stats,
        'linguistic_features': analysis.linguistic_features,
        'keywords': analysis.keywords,
        'plot_urls': analysis.plot_urls,
        'created_at': analysis.created_at.isoformat()
    })

@app.route('/api/cleanup', methods=['POST'])
def api_cleanup():
    """حذف الصور المحلية القديمة"""
    now = time.time()
    deleted = 0
    folder = app.config['UPLOAD_FOLDER']
    for fname in os.listdir(folder):
        if fname.startswith(('radar_', 'bar_', 'wc_')) and fname.endswith('.png'):
            path = os.path.join(folder, fname)
            if now - os.path.getmtime(path) > 3600:
                try:
                    os.remove(path)
                    deleted += 1
                except Exception as e:
                    logging.error(f"فشل حذف {fname}: {e}")
    return jsonify({'deleted': deleted})

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

# نقطة نهاية للتسجيل (اختياري)
@app.route('/api/register', methods=['POST'])
def api_register():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    if not username or not email or not password:
        return jsonify({'error': 'جميع الحقول مطلوبة'}), 400
    if User.query.filter_by(username=username).first():
        return jsonify({'error': 'اسم المستخدم موجود بالفعل'}), 400
    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'البريد الإلكتروني موجود بالفعل'}), 400
    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    token = user.generate_token()
    return jsonify({'token': token, 'user': {'id': user.id, 'username': user.username, 'email': user.email}})

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return jsonify({'error': 'اسم المستخدم أو كلمة المرور غير صحيحة'}), 401
    token = user.generate_token()
    return jsonify({'token': token, 'user': {'id': user.id, 'username': user.username, 'email': user.email}})

# ====================== إنشاء قاعدة البيانات ======================
@app.cli.command("init-db")
def init_db():
    db.create_all()
    print("✅ قاعدة البيانات مهيأة")

# ====================== تشغيل التطبيق ======================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=app.config.get('DEBUG', False))
