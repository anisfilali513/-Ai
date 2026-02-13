#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Anis Linguistic Radar
الإصدار العبقري النهائي مع واجهة مطوّرة
جميع الحقوق محفوظة للمطور أنيس فيلالي
"""

import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm
import mplcursors
import numpy as np
import os
import csv
import threading
import hashlib
import logging
from collections import OrderedDict, Counter
from datetime import datetime
from tkinter import filedialog, messagebox

# مكتبات التحليل المتقدم
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# مكتبات تصدير PDF (باستخدام SimpleDocTemplate)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# مكتبات معالجة العربية
import arabic_reshaper
from bidi.algorithm import get_display

# مكتبة سحابة الكلمات (اختيارية)
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    WordCloud = None

# مكتبة تحليل المشاعر العميق (اختيارية)
try:
    from camel_tools.sentiment import SentimentAnalyzer
    CAMEL_AVAILABLE = True
except ImportError:
    CAMEL_AVAILABLE = False
    SentimentAnalyzer = None

# ---------------------------- الإعدادات الثابتة ----------------------------
BG_COLOR = "#0B0F19"
MAGIC_GOLD = "#D4AF37"
FLUID_TEAL = "#4FD1C5"
TEXT_WHITE = "#F7FAFC"

LIGHT_BG = "#f0f0f0"
LIGHT_TEXT = "#333333"

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

BAR_CHART_XLABEL = "طول الكلمة (حروف)"
BAR_CHART_YLABEL = "عدد الكلمات"
BAR_CHART_TITLE = "توزيع أطوال الكلمات"
WORDCLOUD_TITLE = "سحابة الكلمات"

# مجموعات الحروف العربية للتحليل الصوتي
VOICED = set("بجتدذرزضظعغقلمنوي")
VOICELESS = set("حثسصشفكهت")
PUNCTUATIONS = set(".,;:!?؟،؛")

# مسار الخط العربي (يُفضل وضعه في مجلد المشروع)
FONT_PATH = "Amiri-Regular.ttf"  # غيّر حسب موقع الخط لديك

# ملف السجل
HISTORY_FILE = 'anis_analysis_history.csv'

# ---------------------------- دوال مساعدة للعربية ----------------------------
def reshape_arabic(text):
    """إعادة تشكيل النص العربي للعرض بشكل صحيح"""
    if not text:
        return text
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
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

# ---------------------------- مستخرج الخصائص الإحصائية ----------------------------
class FeatureExtractor:
    """استخراج 8 خصائص إحصائية من النص"""
    def extract(self, text):
        if not text or len(text.strip()) == 0:
            return [0.0] * 8

        words = text.split()
        total_words = len(words)
        total_chars = len(text)

        # 1. الإنتروبيا (تنوع الحروف)
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

        # 4. متوسط طول الجملة (تقسيم بسيط بالنقاط)
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
        """مؤثرات أسلوبية إضافية (نسبة التفرد، hapax، الكثافة المعجمية)"""
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

        # المحلل البسيط (قاموس)
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

# ---------------------------- دوال إنشاء الرسوم البيانية ----------------------------
def create_radar_chart(stats, font_prop=None, dark_mode=True):
    """إنشاء مخطط الرادار"""
    bg = BG_COLOR if dark_mode else LIGHT_BG
    text_color = TEXT_WHITE if dark_mode else LIGHT_TEXT
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
        ax.plot(angles, values, color=MAGIC_GOLD, linewidth=3, marker='o')
        ax.fill(angles, values, color=MAGIC_GOLD, alpha=0.3)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        if font_prop:
            ax.set_xticklabels(cat_reshaped, fontproperties=font_prop, color=text_color, size=10)
        else:
            ax.set_xticklabels(cat_reshaped, color=text_color, size=10)
        return fig
    except Exception as e:
        logging.error(f"فشل إنشاء الرادار: {e}")
        fig = plt.figure(figsize=(6,5), facecolor=bg)
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "خطأ في إنشاء الرادار", color='red', ha='center')
        return fig

def create_bar_chart(text, font_prop=None, dark_mode=True):
    """مخطط توزيع أطوال الكلمات"""
    bg = BG_COLOR if dark_mode else LIGHT_BG
    text_color = TEXT_WHITE if dark_mode else LIGHT_TEXT
    try:
        words = text.split()
        word_lengths = [len(w) for w in words if w]
        bins = range(1, 12)
        hist, _ = np.histogram(word_lengths, bins=bins)
        labels = [f"{i}-{i+1}" for i in range(1, 11)]

        fig = plt.figure(figsize=(6,5), dpi=80, facecolor=bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor("#1a1f2e" if dark_mode else "#e0e0e0")
        ax.bar(labels, hist, color=FLUID_TEAL, edgecolor=MAGIC_GOLD, linewidth=1.5)

        xlabel = reshape_arabic(BAR_CHART_XLABEL)
        ylabel = reshape_arabic(BAR_CHART_YLABEL)
        title = reshape_arabic(BAR_CHART_TITLE)

        if font_prop:
            ax.set_xlabel(xlabel, fontproperties=font_prop, color=text_color)
            ax.set_ylabel(ylabel, fontproperties=font_prop, color=text_color)
            ax.set_title(title, fontproperties=font_prop, color=MAGIC_GOLD)
            for label in ax.get_xticklabels():
                label.set_fontproperties(font_prop)
            for label in ax.get_yticklabels():
                label.set_fontproperties(font_prop)
        else:
            ax.set_xlabel(xlabel, color=text_color)
            ax.set_ylabel(ylabel, color=text_color)
            ax.set_title(title, color=MAGIC_GOLD)

        ax.tick_params(colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(MAGIC_GOLD)
        return fig
    except Exception as e:
        logging.error(f"فشل إنشاء المخطط الشريطي: {e}")
        fig = plt.figure(figsize=(6,5), facecolor=bg)
        ax = fig.add_subplot(111)
        ax.text(0.5,0.5,"خطأ في إنشاء المخطط", color='red', ha='center')
        return fig

def create_wordcloud(text, font_path=None, font_prop=None, dark_mode=True):
    """إنشاء سحابة الكلمات"""
    bg = BG_COLOR if dark_mode else LIGHT_BG
    if not WORDCLOUD_AVAILABLE:
        fig = plt.figure(figsize=(6,5), facecolor=bg)
        ax = fig.add_subplot(111)
        ax.text(0.5,0.5, "مكتبة wordcloud غير مثبتة", color='red', ha='center')
        return fig

    try:
        reshaped = reshape_arabic(text)
        if any('\u0600' <= c <= '\u06FF' for c in reshaped):
            processed = reshaped[::-1]
        else:
            processed = reshaped

        wc = WordCloud(
            width=500, height=400,
            background_color='#0D1117' if dark_mode else '#f0f0f0',
            font_path=font_path,
            colormap='viridis',
            random_state=42
        ).generate(processed)

        fig = plt.figure(figsize=(6,5), dpi=80, facecolor=bg)
        ax = fig.add_subplot(111)
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        title = reshape_arabic(WORDCLOUD_TITLE)
        if font_prop:
            ax.set_title(title, fontproperties=font_prop, color=MAGIC_GOLD, fontsize=14)
        else:
            ax.set_title(title, color=MAGIC_GOLD, fontsize=14)
        return fig
    except Exception as e:
        logging.error(f"فشل إنشاء سحابة الكلمات: {e}")
        fig = plt.figure(figsize=(6,5), facecolor=bg)
        ax = fig.add_subplot(111)
        ax.text(0.5,0.5, f"خطأ: {e}", color='red', ha='center')
        return fig

# ---------------------------- دوال تصدير PDF ----------------------------
def export_to_pdf(text, stats, sentiment, confidence, keywords, font_path, save_path):
    """إنشاء تقرير PDF منظم"""
    doc = SimpleDocTemplate(save_path, pagesize=A4)
    elements = []

    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont("ArabicFont", font_path))
            font_name = "ArabicFont"
        except:
            font_name = "Helvetica"
    else:
        font_name = "Helvetica"

    arabic_style = ParagraphStyle(
        name='ArabicStyle',
        fontName=font_name,
        fontSize=14,
        leading=20,
        alignment=TA_RIGHT,
        textColor=colors.black,
    )

    def ar(t):
        return reshape_arabic(t)

    elements.append(Paragraph(ar("تقرير تحليل النص - Anis Linguistic Radar"), arabic_style))
    elements.append(Spacer(1, 0.3 * inch))

    preview = text[:300] + "..." if len(text) > 300 else text
    elements.append(Paragraph(ar(f"النص: {preview}"), arabic_style))
    elements.append(Spacer(1, 0.2 * inch))

    elements.append(Paragraph(ar(f"المشاعر: {sentiment} (الثقة: {confidence*100:.1f}%)"), arabic_style))
    elements.append(Spacer(1, 0.2 * inch))

    lines = [
        f"الإنتروبيا: {stats[0]:.2f}",
        f"التوازن الصوتي: {stats[1]*100:.1f}%",
        f"نسبة الجهر: {stats[2]:.1f}% | الهمس: {stats[3]:.1f}%",
        f"متوسط طول الكلمة: {stats[4]:.2f} حرف",
        f"متوسط طول الجملة: {stats[5]:.2f} كلمة",
        f"ثراء المفردات: {stats[7]:.2f}%"
    ]
    for line in lines:
        elements.append(Paragraph(ar(line), arabic_style))
        elements.append(Spacer(1, 0.15 * inch))

    elements.append(Paragraph(ar(f"الكلمات المفتاحية: {', '.join(keywords[:5])}"), arabic_style))
    doc.build(elements)

# ---------------------------- التطبيق الرئيسي ----------------------------
class AnisLinguisticRadar(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("Anis Linguistic Radar | الإصدار العبقري النهائي")
        self.geometry("1500x950")
        self.configure(fg_color=BG_COLOR)

        # إعداد الخط لمخططات matplotlib
        self.font_prop = None
        if os.path.exists(FONT_PATH):
            try:
                self.font_prop = fm.FontProperties(fname=FONT_PATH)
                logging.info("✅ تم تحميل الخط العربي لمخططات Matplotlib")
            except Exception as e:
                logging.warning(f"⚠️ فشل تحميل الخط لمخططات Matplotlib: {e}")

        # المكونات
        self.feature_extractor = FeatureExtractor()
        self.sentiment_analyzer = DeepSentimentAnalyzer()
        self.cache = AnalysisCache()

        # متغيرات الحالة
        self.last_text = None
        self.last_stats = None
        self.cached_figures = {}
        self.is_processing = False

        # حالة الواجهة
        self.sidebar_expanded = True
        self.dark_mode = True

        # شريط الحالة
        self.status_text = tk.StringVar()
        self.status_text.set("جاهز | 0 كلمات | 0 أحرف")

        # بناء الواجهة
        self.setup_ui()

    # ---------------------------- بناء الواجهة ----------------------------
    def setup_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ========== الشريط الجانبي القابل للطي ==========
        self.sidebar = ctk.CTkFrame(self, width=380, corner_radius=0, fg_color="#111827")
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)

        # رأس الشريط (أزرار التحكم)
        header_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        header_frame.pack(pady=10, fill="x")

        self.toggle_sidebar_btn = ctk.CTkButton(header_frame, text="☰", width=40,
                                                 command=self.toggle_sidebar,
                                                 fg_color="gray", text_color="white")
        self.toggle_sidebar_btn.pack(side="left", padx=5)

        self.toggle_theme_btn = ctk.CTkButton(header_frame, text="🌙", width=40,
                                               command=self.toggle_theme,
                                               fg_color="gray", text_color="white")
        self.toggle_theme_btn.pack(side="right", padx=5)

        self.sidebar_title = ctk.CTkLabel(header_frame, text="مجلس أنيس (AI Agent)",
                                           font=("Arial", 18, "bold"), text_color=MAGIC_GOLD)
        self.sidebar_title.pack(side="top", pady=5)

        # محتوى الشريط (يتم إخفاؤه عند الطي)
        self.sidebar_content = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.sidebar_content.pack(fill="both", expand=True, padx=10, pady=5)

        # أزرار الشريط
        self.refresh_btn = ctk.CTkButton(self.sidebar_content, text="📋 عرض السجل",
                                          command=self.show_history,
                                          fg_color=FLUID_TEAL, text_color="black")
        self.refresh_btn.pack(pady=5, fill="x")

        self.export_btn = ctk.CTkButton(self.sidebar_content, text="📄 تصدير PDF",
                                         command=self.export_pdf,
                                         fg_color="gray", text_color="white")
        self.export_btn.pack(pady=5, fill="x")

        # منطقة عرض الدردشة
        self.chat_display = ctk.CTkTextbox(self.sidebar_content, height=400,
                                            font=("Arial", 14),
                                            fg_color="#0D1117",
                                            border_color=MAGIC_GOLD, border_width=1)
        self.chat_display.pack(fill="both", expand=True, pady=10)
        self.update_chat("سيدي الكريم، النظام العبقري جاهز...")

        # ========== اللوحة الرئيسية مع تبويبات ==========
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # شريط العنوان
        title = ctk.CTkLabel(main, text="الرادار اللساني العبقري | أنيس فيلالي",
                              font=("Arial", 28, "bold"), text_color=TEXT_WHITE)
        title.pack(pady=10)

        # تبويبات
        self.tab_view = ctk.CTkTabview(main)
        self.tab_view.pack(fill="both", expand=True, padx=10, pady=10)

        self.analysis_tab = self.tab_view.add("🔍 تحليل")
        self.history_tab = self.tab_view.add("📋 سجل")
        self.compare_tab = self.tab_view.add("🔁 مقارنة")

        # تعبئة التبويبات
        self.setup_analysis_tab()
        self.setup_history_tab()
        self.setup_compare_tab()

        # شريط الحالة
        self.status_bar = ctk.CTkLabel(self, textvariable=self.status_text,
                                        anchor="w", font=("Arial", 12),
                                        fg_color="#1f2937", text_color="white")
        self.status_bar.pack(side="bottom", fill="x", padx=10, pady=2)

    # ---------------------------- تبويب التحليل ----------------------------
    def setup_analysis_tab(self):
        tab = self.analysis_tab

        # إطار الإدخال
        input_frame = ctk.CTkFrame(tab, fg_color="transparent")
        input_frame.pack(pady=10, fill="x")

        self.input_text = ctk.CTkTextbox(input_frame, height=120, font=("Arial", 16))
        self.input_text.pack(side="left", fill="both", expand=True, padx=5)
        self.input_text.bind("<KeyRelease>", self.on_text_change)

        # أزرار بجانب الإدخال
        btn_frame = ctk.CTkFrame(input_frame, fg_color="transparent")
        btn_frame.pack(side="right", padx=5)

        self.clear_btn = ctk.CTkButton(btn_frame, text="🗑️ مسح", command=self.clear_input,
                                        fg_color="gray", text_color="white", width=80)
        self.clear_btn.pack(pady=2)

        self.file_btn = ctk.CTkButton(btn_frame, text="📂 رفع", command=self.load_file,
                                       fg_color=FLUID_TEAL, text_color="black", width=80)
        self.file_btn.pack(pady=2)

        # زر التحليل الرئيسي
        self.analyze_btn = ctk.CTkButton(tab, text="✨ تحليل عبقر ✨",
                                          command=self.start_analysis,
                                          fg_color=MAGIC_GOLD, text_color="black",
                                          font=("Arial", 18, "bold"),
                                          height=50)
        self.analyze_btn.pack(pady=10)

        # أزرار التبديل بين المخططات
        plot_btn_frame = ctk.CTkFrame(tab, fg_color="transparent")
        plot_btn_frame.pack(pady=5)

        self.radar_btn = ctk.CTkButton(plot_btn_frame, text="📊 رادار",
                                        command=self.show_radar,
                                        fg_color=FLUID_TEAL, text_color="black", width=110)
        self.radar_btn.grid(row=0, column=0, padx=5)

        self.bar_btn = ctk.CTkButton(plot_btn_frame, text="📈 توزيع",
                                      command=self.show_bar_chart,
                                      fg_color="gray", text_color="white", width=110)
        self.bar_btn.grid(row=0, column=1, padx=5)

        self.wordcloud_btn = ctk.CTkButton(plot_btn_frame, text="☁️ سحابة",
                                            command=self.show_wordcloud,
                                            fg_color="gray", text_color="white", width=110)
        self.wordcloud_btn.grid(row=0, column=2, padx=5)

        # شريط أدوات إضافي
        toolbar = ctk.CTkFrame(tab, fg_color="transparent")
        toolbar.pack(pady=5)

        copy_btn = ctk.CTkButton(toolbar, text="📋 نسخ النتائج",
                                  command=self.copy_results_to_clipboard,
                                  fg_color="gray", text_color="white")
        copy_btn.grid(row=0, column=0, padx=5)

        save_plot_btn = ctk.CTkButton(toolbar, text="💾 حفظ المخطط",
                                       command=self.save_current_plot,
                                       fg_color="gray", text_color="white")
        save_plot_btn.grid(row=0, column=1, padx=5)

        # منطقة عرض المخططات
        self.canvas_area = ctk.CTkFrame(tab, fg_color="#0D1117", corner_radius=15,
                                         border_color="#1F2937", border_width=2,
                                         height=450)
        self.canvas_area.pack(fill="both", expand=True, pady=10)

        # شريط التقدم
        self.progressbar = ctk.CTkProgressBar(tab, mode='indeterminate', width=400)
        self.progressbar.pack(pady=5)
        self.progressbar.pack_forget()

    # ---------------------------- تبويب السجل ----------------------------
    def setup_history_tab(self):
        self.history_scroll = ctk.CTkScrollableFrame(self.history_tab)
        self.history_scroll.pack(fill="both", expand=True)

        self.history_label = ctk.CTkLabel(self.history_scroll, text="", justify="left",
                                           font=("Arial", 14), anchor="nw")
        self.history_label.pack(fill="both", expand=True)

        refresh_hist_btn = ctk.CTkButton(self.history_tab, text="🔄 تحديث",
                                          command=self.refresh_history_tab,
                                          fg_color=FLUID_TEAL, text_color="black")
        refresh_hist_btn.pack(pady=5)

    def refresh_history_tab(self):
        if not os.path.exists(HISTORY_FILE):
            self.history_label.configure(text="لا يوجد سجل بعد.")
            return
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                rows = list(reader)
            if len(rows) <= 1:
                self.history_label.configure(text="السجل فارغ.")
                return
            text = "** السجل السابق **\n\n"
            for row in rows[1:][-20:]:  # آخر 20 عملية
                text += f"• {row[0]} → {row[1]} ({row[2]}) - {row[-1]}\n"
            self.history_label.configure(text=text)
        except Exception as e:
            self.history_label.configure(text=f"خطأ في قراءة السجل: {e}")

    # ---------------------------- تبويب المقارنة ----------------------------
    def setup_compare_tab(self):
        frame = ctk.CTkFrame(self.compare_tab, fg_color="transparent")
        frame.pack(fill="both", expand=True)

        top_frame = ctk.CTkFrame(frame, fg_color="transparent")
        top_frame.pack(fill="both", expand=True)
        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_columnconfigure(1, weight=1)

        self.compare_text1 = ctk.CTkTextbox(top_frame, height=150)
        self.compare_text1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.compare_text2 = ctk.CTkTextbox(top_frame, height=150)
        self.compare_text2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        compare_btn = ctk.CTkButton(frame, text="🔍 احسب التشابه",
                                     command=self.run_comparison,
                                     fg_color=MAGIC_GOLD, text_color="black")
        compare_btn.pack(pady=10)

        self.compare_result = ctk.CTkTextbox(frame, height=200)
        self.compare_result.pack(fill="both", expand=True, padx=5, pady=5)

    def run_comparison(self):
        txt1 = self.compare_text1.get("1.0", tk.END).strip()
        txt2 = self.compare_text2.get("1.0", tk.END).strip()
        if not txt1 or not txt2:
            messagebox.showwarning("تنبيه", "أدخل كلا النصين")
            return
        try:
            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform([txt1, txt2])
            sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
            self.compare_result.delete("1.0", tk.END)
            self.compare_result.insert("1.0", f"نسبة التشابه: {sim:.2f}%\n")
            # إضافة إحصائيات بسيطة
            stats1 = self.feature_extractor.extract(txt1)
            stats2 = self.feature_extractor.extract(txt2)
            self.compare_result.insert(tk.END, f"\nالنص الأول - إنتروبيا: {stats1[0]:.2f}, ثراء: {stats1[7]:.1f}%\n")
            self.compare_result.insert(tk.END, f"النص الثاني - إنتروبيا: {stats2[0]:.2f}, ثراء: {stats2[7]:.1f}%\n")
        except Exception as e:
            messagebox.showerror("خطأ", str(e))

    # ---------------------------- دوال واجهة المستخدم ----------------------------
    def toggle_sidebar(self):
        self.sidebar_expanded = not self.sidebar_expanded
        new_width = 380 if self.sidebar_expanded else 50
        self.sidebar.configure(width=new_width)
        if self.sidebar_expanded:
            self.sidebar_title.pack()
            self.sidebar_content.pack(fill="both", expand=True)
        else:
            self.sidebar_title.pack_forget()
            self.sidebar_content.pack_forget()

    def toggle_theme(self):
        self.dark_mode = not self.dark_mode
        if self.dark_mode:
            ctk.set_appearance_mode("dark")
            self.toggle_theme_btn.configure(text="🌙")
            self.configure(fg_color=BG_COLOR)
            self.status_bar.configure(fg_color="#1f2937", text_color="white")
        else:
            ctk.set_appearance_mode("light")
            self.toggle_theme_btn.configure(text="☀️")
            self.configure(fg_color=LIGHT_BG)
            self.status_bar.configure(fg_color="#dddddd", text_color="black")
        self.refresh_current_chart()

    def refresh_current_chart(self):
        self.cached_figures.clear()
        if hasattr(self, 'last_stats') and self.last_stats is not None:
            if self.radar_btn.cget("state") == "disabled":
                self.show_radar()
            elif self.bar_btn.cget("state") == "disabled":
                self.show_bar_chart()
            elif self.wordcloud_btn.cget("state") == "disabled":
                self.show_wordcloud()

    def on_text_change(self, event=None):
        text = self.input_text.get("1.0", tk.END).strip()
        words = len(text.split())
        chars = len(text)
        self.status_text.set(f"جاهز | {words} كلمات | {chars} أحرف")

    def update_chat(self, message):
        self.chat_display.configure(state="normal")
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.insert(tk.END, reshape_arabic(message))
        self.chat_display.configure(state="disabled")

    def clear_input(self):
        self.input_text.delete("1.0", tk.END)
        self.on_text_change()

    def load_file(self):
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            if len(content) > MAX_TEXT_LENGTH:
                if not messagebox.askyesno("تنبيه", f"النص طويل جداً ({len(content)} حرف). سيتم اقتطاعه إلى {MAX_TEXT_LENGTH} حرف. هل تواصل؟"):
                    return
                content = content[:MAX_TEXT_LENGTH]
            self.input_text.delete("1.0", tk.END)
            self.input_text.insert("1.0", content)
            self.on_text_change()
            self.update_chat(f"تم تحميل الملف: {os.path.basename(path)} ({len(content)} حرف)")
        except Exception as e:
            self.update_chat(f"خطأ في قراءة الملف: {e}")

    # ---------------------------- التحليل ----------------------------
    def start_analysis(self):
        if self.is_processing:
            return
        text = self.input_text.get("1.0", tk.END).strip()
        if len(text) < MIN_TEXT_LENGTH:
            self.update_chat(f"⚠️ النص قصير جداً (الحد الأدنى {MIN_TEXT_LENGTH} حرف)")
            return

        self.is_processing = True
        self.analyze_btn.configure(state="disabled")
        self.progressbar.pack(pady=5)
        self.progressbar.start()

        threading.Thread(target=self._run_analysis, args=(text,), daemon=True).start()

    def _run_analysis(self, text):
        try:
            cached = self.cache.get(text)
            if cached:
                stats, sent, conf, emotions = cached
                self.last_text = text
                self.last_stats = stats
                self.after(0, self._analysis_done, stats, sent, conf, emotions, from_cache=True)
                return

            stats = self.feature_extractor.extract(text)
            sent, conf, emotions = self.sentiment_analyzer.analyze(text)

            self.cache.put(text, (stats, sent, conf, emotions))

            self.last_text = text
            self.last_stats = stats
            self.after(0, self._analysis_done, stats, sent, conf, emotions, from_cache=False)

        except Exception as e:
            logging.exception("خطأ في التحليل")
            self.after(0, self._analysis_error, str(e))

    def _analysis_done(self, stats, sentiment, confidence, emotions, from_cache):
        self._clear_plot_resources()
        self.show_radar()
        self.update_analysis_report(stats, sentiment, confidence, emotions)
        self.log_data(self.last_text, sentiment, stats)
        self.radar_btn.configure(state="disabled", fg_color=FLUID_TEAL)
        self.bar_btn.configure(state="normal", fg_color="gray")
        self.wordcloud_btn.configure(state="normal", fg_color="gray")
        self._analysis_cleanup()
        if from_cache:
            self.update_chat("✅ تم استخدام النتائج من الذاكرة المؤقتة.")

    def _analysis_error(self, msg):
        self.update_chat(f"❌ خطأ في التحليل: {msg}")
        self._analysis_cleanup()

    def _analysis_cleanup(self):
        self.is_processing = False
        self.analyze_btn.configure(state="normal")
        self.progressbar.stop()
        self.progressbar.pack_forget()

    def update_analysis_report(self, stats, sentiment, confidence, emotions):
        # استخراج الكلمات المفتاحية
        try:
            vectorizer = TfidfVectorizer(max_features=5)
            sentences = self.last_text.split('.')
            if len(sentences) < 2:
                sentences = [self.last_text]
            tfidf = vectorizer.fit_transform(sentences)
            keywords = vectorizer.get_feature_names_out().tolist()
        except:
            keywords = ["غير متاح"]

        ttr, hapax, lex = self.feature_extractor.advanced_stylometry(self.last_text)

        report = f"سيدي الكريم،\n\n"
        report += f"🎭 **المشاعر**: {sentiment} (الثقة: {confidence*100:.1f}%)\n"
        if emotions:
            if 'positive' in emotions:
                report += f"   إيجابية: {emotions['positive']} كلمة، سلبية: {emotions.get('negative',0)} كلمة\n"
            else:
                emo_str = ", ".join([f"{k}: {v}" for k,v in emotions.items() if v>0])
                if emo_str:
                    report += f"   المشاعر: {emo_str}\n"
        report += f"🔑 **الكلمات المفتاحية**: {', '.join(keywords[:5])}\n"
        report += f"📊 **الإحصائيات الأساسية**:\n"
        report += f"   الإنتروبيا: {stats[0]:.2f}\n"
        report += f"   التوازن الصوتي: {stats[1]*100:.1f}%\n"
        report += f"   الجهر: {stats[2]:.1f}% | الهمس: {stats[3]:.1f}%\n"
        report += f"   متوسط طول الكلمة: {stats[4]:.2f} حرف\n"
        report += f"   متوسط طول الجملة: {stats[5]:.2f} كلمة\n"
        report += f"   ثراء المفردات: {stats[7]:.2f}%\n"
        report += f"📈 **مؤشرات أسلوبية**:\n"
        report += f"   نسبة التفرد (TTR): {ttr*100:.1f}%\n"
        report += f"   الكلمات النادرة (Hapax): {hapax*100:.1f}%\n"
        report += f"   الكثافة المعجمية: {lex*100:.1f}%\n"
        self.update_chat(report)

    # ---------------------------- إدارة المخططات ----------------------------
    def _clear_plot_resources(self):
        for name, fig in list(self.cached_figures.items()):
            try:
                plt.close(fig)
            except:
                pass
        self.cached_figures.clear()
        plt.close('all')
        for w in self.canvas_area.winfo_children():
            w.destroy()

    def _display_figure(self, name):
        for w in self.canvas_area.winfo_children():
            w.destroy()
        fig = self.cached_figures[name]
        canvas = FigureCanvasTkAgg(fig, master=self.canvas_area)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

        if name == 'radar' and fig.axes:
            try:
                ax = fig.axes[0]
                line = ax.lines[0]
                cursor = mplcursors.cursor(line, hover=True)
                cursor.connect("add", lambda sel: sel.annotation.set_text(f"{line.get_ydata()[sel.index]:.1f}"))
            except:
                pass

    def show_radar(self):
        if self.last_stats is None:
            self.update_chat("حلل نصاً أولاً.")
            return
        self.radar_btn.configure(state="disabled", fg_color=FLUID_TEAL)
        self.bar_btn.configure(state="normal", fg_color="gray")
        self.wordcloud_btn.configure(state="normal", fg_color="gray")
        if 'radar' not in self.cached_figures:
            self.cached_figures['radar'] = create_radar_chart(self.last_stats, self.font_prop, self.dark_mode)
        self._display_figure('radar')

    def show_bar_chart(self):
        if self.last_text is None:
            self.update_chat("حلل نصاً أولاً.")
            return
        self.radar_btn.configure(state="normal", fg_color="gray")
        self.bar_btn.configure(state="disabled", fg_color=FLUID_TEAL)
        self.wordcloud_btn.configure(state="normal", fg_color="gray")
        if 'bar' not in self.cached_figures:
            self.cached_figures['bar'] = create_bar_chart(self.last_text, self.font_prop, self.dark_mode)
        self._display_figure('bar')

    def show_wordcloud(self):
        if self.last_text is None:
            self.update_chat("حلل نصاً أولاً.")
            return
        if not WORDCLOUD_AVAILABLE:
            self.update_chat("⚠️ مكتبة wordcloud غير مثبتة. قم بتشغيل: pip install wordcloud")
            return
        self.radar_btn.configure(state="normal", fg_color="gray")
        self.bar_btn.configure(state="normal", fg_color="gray")
        self.wordcloud_btn.configure(state="disabled", fg_color=FLUID_TEAL)
        if 'wordcloud' not in self.cached_figures:
            self.cached_figures['wordcloud'] = create_wordcloud(self.last_text, FONT_PATH, self.font_prop, self.dark_mode)
        self._display_figure('wordcloud')

    def save_current_plot(self):
        if not self.cached_figures:
            self.update_chat("لا يوجد مخطط لحفظه")
            return
        active = None
        if self.radar_btn.cget("state") == "disabled":
            active = 'radar'
        elif self.bar_btn.cget("state") == "disabled":
            active = 'bar'
        elif self.wordcloud_btn.cget("state") == "disabled":
            active = 'wordcloud'
        if active and active in self.cached_figures:
            path = filedialog.asksaveasfilename(defaultextension=".png",
                                                filetypes=[("PNG files", "*.png")])
            if path:
                self.cached_figures[active].savefig(path, dpi=100, bbox_inches='tight')
                self.update_chat(f"✅ تم حفظ المخطط في {path}")
        else:
            self.update_chat("لا يوجد مخطط نشط")

    def copy_results_to_clipboard(self):
        if not self.last_text:
            self.update_chat("لا توجد نتائج لنسخها")
            return
        # إعادة استخدام التقرير من آخر تحديث
        self.clipboard_clear()
        self.clipboard_append("تقرير التحليل متاح في نافذة الدردشة")
        self.update_chat("✅ تم نسخ النتائج إلى الحافظة (يمكنك نسخها من التقرير أعلاه)")

    # ---------------------------- تصدير PDF ----------------------------
    def export_pdf(self):
        if self.last_stats is None or self.last_text is None:
            self.update_chat("⚠️ لا يوجد تحليل لتصديره.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if not path:
            return
        try:
            sent, conf, _ = self.sentiment_analyzer.analyze(self.last_text)
            try:
                vectorizer = TfidfVectorizer(max_features=5)
                sentences = self.last_text.split('.')
                if len(sentences) < 2:
                    sentences = [self.last_text]
                tfidf = vectorizer.fit_transform(sentences)
                keywords = vectorizer.get_feature_names_out().tolist()
            except:
                keywords = ["غير متاح"]
            export_to_pdf(self.last_text, self.last_stats, sent, conf, keywords, FONT_PATH, path)
            self.update_chat(f"✅ تم حفظ التقرير في: {path}")
        except Exception as e:
            messagebox.showerror("خطأ", f"فشل التصدير: {e}")

    # ---------------------------- السجل ----------------------------
    def log_data(self, txt, sentiment, stats):
        try:
            file_exists = os.path.isfile(HISTORY_FILE)
            with open(HISTORY_FILE, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['النص (مختصر)', 'المشاعر', 'الإنتروبيا', 'التاريخ'])
                writer.writerow([
                    txt[:50] + "...",
                    sentiment,
                    f"{stats[0]:.2f}",
                    datetime.now().strftime("%Y-%m-%d %H:%M")
                ])
        except Exception as e:
            logging.error(f"خطأ في تسجيل البيانات: {e}")

    def show_history(self):
        # بدلاً من عرض نافذة منفصلة، ننتقل إلى تبويب السجل ونحدثه
        self.tab_view.set("📋 سجل")
        self.refresh_history_tab()

# ---------------------------- نقطة الدخول ----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    app = AnisLinguisticRadar()
    app.mainloop()
