#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anis Linguistic Radar - Web Version (النسخة النهائية)
Flask-based web application for Arabic text analysis
جميع الحقوق محفوظة للمطور أنيس فيلالي
"""

from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')  # استخدام الخلفية غير التفاعلية (للسيرفر)
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
import csv
import hashlib
import logging
import uuid
from collections import Counter, OrderedDict
from datetime import datetime
import arabic_reshaper
from bidi.algorithm import get_display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# محاولة استيراد wordcloud (اختياري)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    WordCloud = None

# محاولة استيراد camel-tools (اختياري)
try:
    from camel_tools.sentiment import SentimentAnalyzer
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    SentimentAnalyzer = None

# ---------------------------- الإعدادات الثابتة ----------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'anis-secret-key-change-in-production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB حد للملفات

# إعداد التسجيل
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# المسارات
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_IMAGES_DIR = os.path.join(BASE_DIR, 'static', 'images')
HISTORY_FILE = os.path.join(BASE_DIR, 'history.csv')
FONT_PATH = os.path.join(BASE_DIR, 'Amiri-Regular.ttf')

# إنشاء مجلد الصور إذا لم يكن موجودًا
os.makedirs(STATIC_IMAGES_DIR, exist_ok=True)

# إعداد الخط العربي لمخططات matplotlib
font_prop = None
if os.path.exists(FONT_PATH):
    try:
        font_prop = fm.FontProperties(fname=FONT_PATH)
        logging.info("✅ تم تحميل الخط العربي لمخططات Matplotlib")
    except Exception as e:
        logging.warning(f"⚠️ فشل تحميل الخط لمخططات Matplotlib: {e}")
else:
    logging.warning("⚠️ خط Amiri غير موجود. قد لا تظهر العربية بشكل صحيح في المخططات.")

# ثوابت التحليل
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 20000
RADAR_CATEGORIES = [
    "الإنتروبيا",
    "التوازن الصوتي",
    "الجهر",
    "الهمس",
    "طول الكلمة",
    "ثراء المفردات"
]

# مجموعات الحروف العربية
VOICED = set("بجتدذرزضظعغقلمنوي")
VOICELESS = set("حثسصشفكهت")
PUNCTUATIONS = set(".,;:!?؟،؛")

# ---------------------------- دوال مساعدة للعربية ----------------------------
def reshape_arabic(text):
    """إعادة تشكيل النص العربي للعرض بشكل صحيح"""
    if not text:
        return text
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception as e:
        logging.error(f"خطأ في reshape: {e}")
        return text

# ---------------------------- مدير التخزين المؤقت (Cache) ----------------------------
class AnalysisCache:
    """تخزين مؤقت لنتائج التحليل باستخدام hash النص كمفتاح"""
    def __init__(self, max_size=50):
        self.cache = OrderedDict()
        self.max_size = max_size

    def _hash(self, text):
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def get(self, text):
        key = self._hash(text)
        if key in self.cache:
            self.cache.move_to_end(key)
            logging.info("✅ تم استرجاع النتائج من الكاش")
            return self.cache[key]
        return None

    def put(self, text, result):
        key = self._hash(text)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = result
        logging.info("📦 تم تخزين النتائج في الكاش")

cache = AnalysisCache()

# ---------------------------- مستخرج الخصائص الإحصائية ----------------------------
class FeatureExtractor:
    def extract(self, text):
        if not text or len(text.strip()) == 0:
            return [0.0] * 8

        words = text.split()
        total_words = len(words)
        total_chars = len(text)

        # 1. الإنتروبيا
        char_counts = Counter(text)
        entropy = -sum((count/total_chars) * np.log2(count/total_chars) for count in char_counts.values()) if total_chars else 0

        # 2. التوازن الصوتي
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

        # 3. متوسط طول الكلمة
        avg_word = np.mean([len(w) for w in words]) if words else 0.0

        # 4. متوسط طول الجملة
        sentences = [s.strip() for s in text.replace('!', '.').replace('؟', '.').replace('،', '.').split('.') if s.strip()]
        avg_sentence = total_words / len(sentences) if sentences else total_words

        # 5. ثراء المفردات
        unique_words = len(set(words))
        richness = unique_words / total_words if total_words else 0.0

        # 6. نسبة علامات الترقيم
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

    def advanced_stylometry(self, text):
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

feature_extractor = FeatureExtractor()

# ---------------------------- محلل المشاعر ----------------------------
class DeepSentimentAnalyzer:
    def __init__(self):
        self.analyzer = None
        if CAMEL_AVAILABLE:
            try:
                self.analyzer = SentimentAnalyzer.pretrained()
                logging.info("✅ تم تحميل نموذج المشاعر من camel-tools")
            except Exception as e:
                logging.warning(f"⚠️ فشل تحميل نموذج camel-tools: {e}")
        else:
            logging.info("⚠️ camel-tools غير مثبت، سيتم استخدام المحلل البسيط")

    def analyze(self, text):
        if self.analyzer is not None:
            try:
                result = self.analyzer.predict([text[:512]])[0]
                confidence = 0.85  # تقديري
                emotions = self._extract_emotions(text)
                return result, confidence, emotions
            except Exception as e:
                logging.error(f"خطأ في التحليل العميق: {e}")

        # المحلل البسيط
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

sentiment_analyzer = DeepSentimentAnalyzer()

# ---------------------------- دوال إنشاء الرسوم البيانية ----------------------------
def create_radar_chart(stats, filename, dark_mode=True):
    """إنشاء مخطط الرادار وحفظه في الملف المحدد"""
    bg = "#0B0F19" if dark_mode else "#f0f0f0"
    text_color = "#F7FAFC" if dark_mode else "#333333"
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

        cat_reshaped = [reshape_arabic(cat) for cat in RADAR_CATEGORIES]

        fig = plt.figure(figsize=(6, 5), dpi=80, facecolor=bg)
        ax = fig.add_subplot(111, polar=True)
        ax.set_facecolor("#1a1f2e" if dark_mode else "#e0e0e0")
        ax.plot(angles, values, color="#D4AF37", linewidth=3, marker='o')
        ax.fill(angles, values, color="#D4AF37", alpha=0.3)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        if font_prop:
            ax.set_xticklabels(cat_reshaped, fontproperties=font_prop, color=text_color, size=10)
        else:
            ax.set_xticklabels(cat_reshaped, color=text_color, size=10)
        fig.savefig(filename, bbox_inches='tight', facecolor=bg)
        plt.close(fig)
        return True
    except Exception as e:
        logging.error(f"فشل إنشاء الرادار: {e}")
        return False

def create_bar_chart(text, filename, dark_mode=True):
    """مخطط توزيع أطوال الكلمات"""
    bg = "#0B0F19" if dark_mode else "#f0f0f0"
    text_color = "#F7FAFC" if dark_mode else "#333333"
    try:
        words = text.split()
        word_lengths = [len(w) for w in words if w]
        bins = range(1, 12)
        hist, _ = np.histogram(word_lengths, bins=bins)
        labels = [f"{i}-{i+1}" for i in range(1, 11)]

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
            for label in ax.get_xticklabels():
                label.set_fontproperties(font_prop)
            for label in ax.get_yticklabels():
                label.set_fontproperties(font_prop)
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
    """إنشاء سحابة الكلمات"""
    bg = "#0B0F19" if dark_mode else "#f0f0f0"
    if not WORDCLOUD_AVAILABLE:
        # إنشاء صورة خطأ بدلاً من إرجاع خطأ
        fig = plt.figure(figsize=(6,5), facecolor=bg)
        ax = fig.add_subplot(111)
        ax.text(0.5,0.5, "مكتبة wordcloud غير مثبتة", color='red', ha='center')
        fig.savefig(filename, bbox_inches='tight', facecolor=bg)
        plt.close(fig)
        return True

    try:
        reshaped = reshape_arabic(text)
        # بعض المعالجة الإضافية للعربية في wordcloud
        if any('\u0600' <= c <= '\u06FF' for c in reshaped):
            processed = reshaped[::-1]  # قد تحتاج لتعديل حسب اتجاه النص
        else:
            processed = reshaped

        wc = WordCloud(
            width=500, height=400,
            background_color='#0D1117' if dark_mode else '#f0f0f0',
            font_path=FONT_PATH if os.path.exists(FONT_PATH) else None,
            colormap='viridis',
            random_state=42
        ).generate(processed)

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
        ax.text(0.5,0.5, f"خطأ: {e}", color='red', ha='center')
        fig.savefig(filename, bbox_inches='tight', facecolor=bg)
        plt.close(fig)
        return True

# ---------------------------- دوال CSV ----------------------------
def log_to_csv(text, sentiment, stats):
    """تسجيل عملية تحليل في ملف CSV"""
    try:
        file_exists = os.path.isfile(HISTORY_FILE)
        with open(HISTORY_FILE, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['النص (مختصر)', 'المشاعر', 'الإنتروبيا', 'التاريخ'])
            writer.writerow([
                text[:50] + "...",
                sentiment,
                f"{stats[0]:.2f}",
                datetime.now().strftime("%Y-%m-%d %H:%M")
            ])
    except Exception as e:
        logging.error(f"خطأ في تسجيل البيانات: {e}")

def read_history(limit=50):
    """قراءة آخر limit سجل من ملف CSV"""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)
        if len(rows) <= 1:
            return []
        # نريد آخر limit صف (بدون العنوان)
        return rows[1:][-limit:]
    except Exception as e:
        logging.error(f"خطأ في قراءة السجل: {e}")
        return []

# ---------------------------- نقاط نهاية Flask ----------------------------

@app.route('/')
def index():
    """الصفحة الرئيسية"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """تحليل النص المرسل"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'لا يوجد نص'}), 400

    text = data['text'].strip()
    if len(text) < MIN_TEXT_LENGTH:
        return jsonify({'error': f'النص قصير جداً (الحد الأدنى {MIN_TEXT_LENGTH} حرف)'}), 400
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]  # اقتطاع

    # التحقق من الكاش
    cached = cache.get(text)
    if cached:
        stats, sentiment, confidence, emotions = cached
        from_cache = True
    else:
        stats = feature_extractor.extract(text)
        sentiment, confidence, emotions = sentiment_analyzer.analyze(text)
        cache.put(text, (stats, sentiment, confidence, emotions))
        from_cache = False

    # إنشاء أسماء ملفات فريدة للرسوم البيانية
    plot_id = str(uuid.uuid4())
    radar_filename = os.path.join(STATIC_IMAGES_DIR, f'radar_{plot_id}.png')
    bar_filename = os.path.join(STATIC_IMAGES_DIR, f'bar_{plot_id}.png')
    wc_filename = os.path.join(STATIC_IMAGES_DIR, f'wc_{plot_id}.png')

    # إنشاء الرسوم
    dark_mode = data.get('dark_mode', True)  # يمكن تمريرها من الواجهة
    radar_ok = create_radar_chart(stats, radar_filename, dark_mode)
    bar_ok = create_bar_chart(text, bar_filename, dark_mode)
    wc_ok = create_wordcloud(text, wc_filename, dark_mode)

    # استخراج الكلمات المفتاحية
    try:
        vectorizer = TfidfVectorizer(max_features=5)
        sentences = text.split('.')
        if len(sentences) < 2:
            sentences = [text]
        tfidf = vectorizer.fit_transform(sentences)
        keywords = vectorizer.get_feature_names_out().tolist()
    except Exception as e:
        keywords = ["غير متاح"]

    # مؤشرات أسلوبية متقدمة
    ttr, hapax, lex = feature_extractor.advanced_stylometry(text)

    # تسجيل في CSV
    log_to_csv(text, sentiment, stats)

    # تجهيز مسارات الصور (نسبية للمسار الثابت)
    base_url = '/static/images/'
    response = {
        'success': True,
        'from_cache': from_cache,
        'text_preview': text[:200] + '...' if len(text) > 200 else text,
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
        'advanced': {
            'ttr': round(ttr*100, 1),
            'hapax': round(hapax*100, 1),
            'lexical_density': round(lex*100, 1)
        },
        'keywords': keywords[:5],
        'plots': {
            'radar': base_url + f'radar_{plot_id}.png' if radar_ok else None,
            'bar': base_url + f'bar_{plot_id}.png' if bar_ok else None,
            'wordcloud': base_url + f'wc_{plot_id}.png' if wc_ok else None
        }
    }
    return jsonify(response)

@app.route('/compare', methods=['POST'])
def compare():
    """مقارنة بين نصين"""
    data = request.get_json()
    text1 = data.get('text1', '').strip()
    text2 = data.get('text2', '').strip()
    if not text1 or not text2:
        return jsonify({'error': 'الرجاء إدخال كلا النصين'}), 400

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
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def history():
    """إرجاع آخر عمليات التحليل من CSV"""
    records = read_history(50)
    # تنسيق للعرض
    history_list = []
    for row in records:
        if len(row) >= 4:
            history_list.append({
                'text': row[0],
                'sentiment': row[1],
                'entropy': row[2],
                'time': row[3]
            })
    return jsonify(history_list)

@app.route('/cleanup_images', methods=['POST'])
def cleanup_images():
    """حذف الصور القديمة (يمكن استدعاؤها من الواجهة أو بشكل دوري)"""
    # حذف الصور الأقدم من ساعة واحدة
    import time
    now = time.time()
    deleted = 0
    for fname in os.listdir(STATIC_IMAGES_DIR):
        if fname.startswith(('radar_', 'bar_', 'wc_')) and fname.endswith('.png'):
            path = os.path.join(STATIC_IMAGES_DIR, fname)
            if now - os.path.getmtime(path) > 3600:  # أقدم من ساعة
                os.remove(path)
                deleted += 1
    return jsonify({'deleted': deleted})

# ---------------------------- تشغيل التطبيق ----------------------------
if __name__ == '__main__':
    # تشغيل الخادم محلياً (يمكن تغيير host إلى '0.0.0.0' للوصول من الشبكة المحلية)
    app.run(debug=True, host='0.0.0.0', port=5000)
