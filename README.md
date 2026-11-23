# LearnSphere — Learner Portal + Platform Intelligence

**نسخة محدثة من موقع LearnSphere، مبنية على كود Streamlit الأصلي ونتائج Extra Trees / SHAP / LIME المحفوظة.**

لا تحتاج تسجيل مستخدم، ولا تعيد حساب أي توصيات أو تفسيرات. كل زائر يختار **Learner ID من قائمة Dropdown قابلة للبحث**. عند ربط بيانات الدراسة كاملة ستظهر نتائج **177,247 توصية** (بحد أقصى 3 لكل متعلم) بعد التحقق من ملفاتها.

## شكل الموقع الجديد

### 1. Explore Courses — واجهة المتعلم

- اختيار Learner ID من القائمة الجانبية، مثل `Learner 00001`، من دون حساب أو تسجيل دخول.
- **My previous courses**: المقررات التاريخية السابقة للمتعلم، إذا كان `history.csv` أو `history_display.csv` متاحًا ومصرحًا بعرضه. لو غير متاح ستظهر رسالة بدل بيانات مخترعة.
- **Recommended for you**: أفضل **3 كورسات فقط** بنفس درجات الترتيب المحفوظة، مع سبب ترشيح *مبسّط وواضح داخل كل كارت* من SHAP المحفوظ الحقيقي. لا تُعرض هنا جداول SHAP/LIME أو Jaccard أو Fidelity.
- لا نفترض معرفة رغبات المتعلم الشخصية ولا ندّعي أن التوصية تضمن نجاحه.

### 2. Platform Intelligence — واجهة المنصة والبحث

للمتعلم المختار من نفس قائمة الـIDs وأحد كورساته الثلاثة، تحتوي على أربع تبويبات:

1. **Personalized Strategy**: استراتيجية تسويقية *مقترحة لهذا المتعلم*، تتحدد بشفافية من مجموع القيم المطلقة لأهم SHAP features المرتبطة بتاريخ الاختيار، توقيت الاختيار، أو تنوع الاختيار **عبر توصياته الثلاث**. تظهر الخصائص الداعمة وقاعدة الاختيار؛ ليست تجربة تثبت نجاح الاستراتيجية.
2. **Consumer Behavior**: ربط مفاهيمي بين **Course Participation / Behavioral Engagement / Course Exploration** والمؤشرات المتاحة. لا تُعرض المقاييس اللاحقة لبدء المقرر إلا إذا أمكن تدقيق التواريخ. لا تُنسب أي قيم SHAP صناعية لمؤشرات Engagement/Exploration.
3. **SHAP vs LIME**: رسمان وجداول Top-5/Top-10 لنفس التوصية، إشارة الخصائص، Jaccard، اتجاه الاتفاق وLIME Fidelity R². **اختلاف الطريقتين مطلوب للتحليل وليس شيئًا ينبغي تصحيحه**.
4. **Cohort Analytics**: الإحصائيات المجمعة الموثقة من التفسيرات المخزنة، ومقارنات حسب المقرر ودرجة LIME Fidelity.

### 3. Research Methods

شرح هدف التنبؤ التاريخي خلال 1–28 أغسطس 2013، وقيود Extra Trees وSHAP وLIME، وأبعاد السلوك، وطبيعة الاستراتيجيات البحثية، والخصوصية.

## 1. فك الضغط ونقل بياناتك الحالية — Windows

فكّي الضغط في مكان جديد مثلاً:

`D:\research\test\LearnSphere_v2\LearnSphere_Student_Platform_v2`

افتحي **Anaconda PowerShell** ثم نفّذي كل سطر وحده (Enter بعد كل أمر):

```powershell
cd "D:\research\test\LearnSphere_v2\LearnSphere_Student_Platform_v2"
python -m pip install -r requirements.txt
```

أنتِ سبق ونسختي بياناتك إلى الموقع الحالي. لننسخها إلى النسخة الجديدة دون إعادة SHAP/LIME:

```powershell
python import_saved_data.py --source "D:\research\test\LearnSphere_Public_Ready\LearnSphere_Public_Ready\data" --include-history --include-behavior
```

هذا الأمر ينسخ: `recommendations.csv` والـ12 ملف SQLite، مع `history.csv` و`behavior.csv` **إن وجدا**، للاستخدام المحلي. لو لم تجدي ملفًا اختياريًا فسيظل الموقع شغالًا، مع رسالة توضّح غياب البيانات.

لو مصدر البيانات ما زال في المشروع الأصلي، بدّلي `--source` إلى:

`D:\research\test\LearnSphere_Deployment\data`

### بنية الملفات بعد النسخ

```text
LearnSphere_Student_Platform_v2/
├── app.py
├── personalization.py
├── local_xai.py
├── import_saved_data.py
├── build_public_analytics.py
├── assemble_public_release.py
├── requirements.txt
├── METHODS.md
├── README.md
├── tests/
└── data/
    ├── recommendations.csv              # 177,247 توصية، ملف مطلوب
    ├── public_analytics.json            # إحصائيات مجمعة، يُنشأ تلقائيًا
    ├── history.csv                      # اختياري: للعرض المحلي أو بعد التصريح
    ├── behavior.csv                     # اختياري، يخضع لمراجعة خصوصية
    └── local_xai/
        ├── xai_batch_001.sqlite
        ├── ...
        └── xai_batch_012.sqlite
```

## 2. تجهيز المؤشرات وتشغيل الموقع

```powershell
python build_public_analytics.py --expected 177247
```

التحقق يعيد **تجميع جداول المقارنة من النتائج المخزنة**؛ لا يفتح الموديل، ولا يعيد التنبؤ أو حساب SHAP/LIME. سيُنشئ `data/public_analytics.json`.

بعد اكتمال الأمر:

```powershell
python -m streamlit run app.py
```

افتحي رابط **http://localhost:8501** وجربي Learner 00001 مع CS50x، ثم تأكدي من ظهور الواجهة المبسطة ولوحة المنصة.

## 3. النشر العام — بعد التصريح والمراجعة

**مهم:** وجود معرف بدل الاسم لا يعني إخفاء الهوية بالكامل. صفحة عامة بدون Login تسمح لأي زائر باستعراض كل الـIDs، وقد يعرّض تواريخ أو تسلسلات اختيار المقررات لإعادة التعرف. كذلك لا بد من التحقق من شروط إعادة نشر بيانات HarvardX/MITx وموافقات المؤسسة. لا تنشري البيانات التفصيلية إلا لو مصرح بها.

لإنشاء نسخة نشر من دون سجلات المقررات السابقة (الخيار الأقل تعرضًا):

```powershell
python assemble_public_release.py --confirm-publication
```

**لو مصرح لك بعرض المقررات السابقة للمتعلمين**، يمكن إدراج نسخة مبسطة من التاريخ حتى 31 يوليو 2013، **من غير تواريخ**:

```powershell
python assemble_public_release.py --confirm-publication --include-history-display
```

الاختيار الأخير يحتاج وجود `data/history.csv` في المشروع المحلي، وينتج `history_display.csv` يحتوي على `userid_DI` و`course_id` فقط (بعد تصفية التواريخ قبل أغسطس). **حتى هذه النسخة ممكن تكون معرِّفة بشكل غير مباشر، فلا تُنشر دون تصريح.**

مجلد النشر الناتج: `LearnSphere_Publish/` وبجانبه `LearnSphere_Publish.zip`. كل الملفات المطلوبة للاستضافة منسوخة، بما فيها `personalization.py` ومجلد SQLite. لن يتم رفع شيء تلقائيًا. إن تجاوز أي ملف حد GitHub المسموح، سيوقف البناء بدل تقديم ملفات ناقصة. عند الموافقة على الإتاحة، ارفعي **محتويات المجلد** إلى مستودع GitHub، وافتحي https://share.streamlit.io واختاري `app.py`.

**External URL الذي يظهر من تشغيل Streamlit على جهازك ليس نشرًا مضمونًا للموقع.** رابط عام مستقر يتطلب منصة استضافة وإتمام الإطلاق.

## 4. اختبارات المصدر (دون لمس بيانات المشروع الفعلية)

```powershell
python -m unittest discover -s tests -v
python -m py_compile app.py personalization.py local_xai.py build_public_analytics.py
```

تستخدم الاختبارات ملف بيانات صناعيًا في مجلد مؤقت، ولا تكتب فوق `data/recommendations.csv` الحقيقية.

## تنبيهات منهجية

- **Model score ≠ calibrated probability**، ونتيجة التوصية ≠ مشاركة فعلية أو شراء.
- **SHAP** مساهمات بالنسبة إلى baseline؛ **LIME** أوزان نموذج تقريبي محلي وليست بالضرورة بنفس وحدات SHAP.
- صف `Other features (aggregated)` يُستخدم لفحص المجموع، ولا يُحسب ضمن ترتيب الخصائص.
- **Fidelity R²** يصف جودة تقريب LIME على عيناته المحلية، وليس دقة النموذج أو صحة SHAP.
- استراتيجيات التسويق هنا توصيات بحثية **rule-based** مشتقة من تفسير النموذج، ولا يجوز وصفها كتأثيرات سببية مثبتة.

للتفاصيل البحثية، راجعي `METHODS.md`.
