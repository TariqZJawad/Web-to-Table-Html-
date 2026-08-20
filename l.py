import psycopg2
import requests
import hidden

# 1. جلب بيانات الاتصال من ملفك
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

# 3. تجهيز الجدول المطلوب (حذفه إن وجد لتجنب التكرار، ثم إنشاؤه)
cur.execute('DROP TABLE IF EXISTS pokeapi CASCADE;')
cur.execute('CREATE TABLE IF NOT EXISTS pokeapi (id INTEGER, body JSONB);')
conn.commit()

# 4. بدء الحلقة التكرارية لجلب أول 100 بوكيمون
print("Starting the data pipeline...")

for i in range(1, 101):
    # تجهيز الرابط الديناميكي لكل دورة
    url = f'https://pokeapi.co/api/v2/pokemon/{i}'
    
    # سحب البيانات من الواجهة
    response = requests.get(url)
    
    # التأكد من أن السحب تم بنجاح (الكود 200)
    if response.status_code == 200:
        # أخذ النص كـ JSON
        text_data = response.text
        
        # إدراج الرقم والنص في قاعدة البيانات
        sql = 'INSERT INTO pokeapi (id, body) VALUES (%s, %s);'
        cur.execute(sql, (i, text_data))
        
    # طباعة مؤشر للمراقبة
    if i % 10 == 0:
        print(f"Loaded {i} documents...")

# 5. حفظ التغييرات النهائية وإغلاق الاتصال
conn.commit()
cur.close()
conn.close()

print("Mission Accomplished: 100 JSON documents loaded into PostgreSQL!")
