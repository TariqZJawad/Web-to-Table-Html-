import psycopg2
import hidden

# 1. جلب بيانات الاتصال من ملف hidden
secrets = hidden.secrets()

# 2. فتح الاتصال بقاعدة البيانات
conn = psycopg2.connect(
    host=secrets['host'],
    port=secrets['port'],
    database=secrets['database'],
    user=secrets['user'],
    password=secrets['pass'],
    connect_timeout=3
)
cur = conn.cursor()

# 3. حذف الجدول القديم وإعادة إنشاء جدول pythonseq الجديد
cur.execute('DROP TABLE IF EXISTS pythonseq CASCADE;')
cur.execute('CREATE TABLE pythonseq (iter INTEGER, val INTEGER);')

# 4. تهيئة الرقم الابتدائي كما هو مطلوب في الواجب
value = 684872

# 5. حلقة التوليد والإدراج (300 مرة)
for i in range(300):
    # رقم الدورة الحالي (يبدأ من 1 بدلاً من 0)
    iteration = i + 1
    
    # إدراج البيانات في الجدول
    cur.execute('INSERT INTO pythonseq (iter, val) VALUES (%s, %s);', (iteration, value))
    
    # تحديث القيمة للدورة القادمة باستخدام المعادلة الرياضية
    value = int((value * 22) / 7) % 1000000

# 6. تثبيت التغييرات (Commit) وحفظها في قاعدة البيانات
conn.commit()

# إغلاق الاتصال
cur.close()
conn.close()

print("Mission Accomplished: 300 pseudo-random numbers have been inserted!")
