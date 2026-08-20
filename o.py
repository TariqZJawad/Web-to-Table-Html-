class PhysicalObject:
    def __init__(self, mass: float, distance: float, time: float):
        """
        دالة الإنشاء: نحدد الصفات الأساسية للجسم عند ولادته
        mass: الكتلة بالكيلوغرام (kg)
        distance: المسافة بالمتر (m)
        time: الزمن بالثواني (s)
        """
        self.mass = mass
        self.distance = distance
        self.time = time

    # استخدام @property لأن السرعة تعتبر "صفة مشتقة" لا تحتاج لأقواس عند استدعائها
    @property
    def velocity(self) -> float:
        """حساب السرعة: المسافة / الزمن"""
        if self.time == 0:
            raise ZeroDivisionError("⚠️ لا يمكن حساب السرعة لأن الزمن يساوي صفر!")
        return self.distance / self.time

    @property
    def acceleration(self) -> float:
        """حساب التعجيل: السرعة / الزمن"""
        if self.time == 0:
            raise ZeroDivisionError("⚠️ لا يمكن حساب التعجيل لأن الزمن يساوي صفر!")
        # نستخدم self.velocity لاستدعاء دالة السرعة المكتوبة في الأعلى
        return self.velocity / self.time

    @property
    def jerk(self) -> float:
        """حساب الـ Jerk (معدل تغير التعجيل): التعجيل / الزمن"""
        if self.time == 0:
            raise ZeroDivisionError("⚠️ لا يمكن حساب الـ Jerk لأن الزمن يساوي صفر!")
        return self.acceleration / self.time

    @property
    def momentum(self) -> float:
        """حساب الزخم: الكتلة × السرعة"""
        return self.mass * self.velocity

    @property
    def kinetic_energy(self) -> float:
        """حساب الطاقة الحركية: 0.5 × الكتلة × مربع السرعة"""
        return 0.5 * self.mass * (self.velocity ** 2)

    def update_time(self, new_time: float):
        """دالة إضافية لتحديث الزمن إذا تغيرت حالة الجسم"""
        if new_time <= 0:
            raise ValueError("الزمن يجب أن يكون أكبر من الصفر.")
        self.time = new_time
if __name__ == "__main__":
    # 1. إنشاء الجسم الفيزيائي (الكتلة=1500، المسافة=100، الزمن=5)
    car = PhysicalObject(mass=1500, distance=100, time=5)
    
    print("--- 📊 البيانات الفيزيائية للسيارة ---")
    
    # 2. استدعاء الدوال (لاحظ استدعاءها بدون أقواس لأننا استخدمنا @property)
    print(f"🏎️ السرعة (Velocity):       {car.velocity:.2f} m/s")
    print(f"📈 التعجيل (Acceleration):  {car.acceleration:.2f} m/s²")
    print(f"⚡ الـ Jerk:                {car.jerk:.2f} m/s³")
    print(f"📦 الزخم (Momentum):        {car.momentum:.2f} kg·m/s")
    print(f"🔥 الطاقة (Kinetic Energy): {car.kinetic_energy:.2f} Joules")
    
    print("\n--------------------------------------\n")
    
    # 3. ماذا لو ضغط السائق على المكابح وتغير الزمن ليصبح 10 ثوانٍ لنفس المسافة؟
    print("🔄 تحديث حالة السيارة (زيادة الزمن إلى 10 ثوانٍ)...")
    car.update_time(10)
    
    print(f"📉 السرعة الجديدة: {car.velocity:.2f} m/s")
    print(f"🔥 الطاقة الجديدة: {car.kinetic_energy:.2f} Joules")
