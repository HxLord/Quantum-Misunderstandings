import pygame
import random
import math
import matplotlib.pyplot as plt

# --- 1. Инициализация систем ---
pygame.init()
WIDTH, HEIGHT = 1350, 850
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Kvantovye Nedorazumeniya: Orbital Shield Pro")
clock = pygame.time.Clock()
shield_lines = []  # Список для хранения объектов вертикальных линий

# Цвета и Шрифты
COLORS = {'bg': (5, 5, 15), 'sun': (255, 200, 0), 'earth': (30, 120, 255), 
          'A': (180, 180, 180), 'B': (0, 200, 255), 'C': (255, 255, 0), 'M': (255, 100, 0), 'X': (255, 0, 100)}
font_s = pygame.font.SysFont("Consolas", 14)
font_m = pygame.font.SysFont("Segoe UI", 18)
font_l = pygame.font.SysFont("Segoe UI", 22, bold=True)

# Глобальные настройки
sim_speed = 1.0
protection_active = True
stats = {'A': 0, 'B': 0, 'C': 0, 'M': 0, 'X': 0, 'data_lost': 0}

# Настройка Matplotlib (График)
plt.ion()
fig, ax_plt = plt.subplots(figsize=(5, 4))
ax_plt.set_facecolor('#0a0a14') # Темный фон
fig.patch.set_facecolor('#0a0a14')

def setup_plot_style():
    """Функция для настройки стиля и зон графика"""
    ax_plt.clear()
    ax_plt.set_yscale('log')
    ax_plt.axhspan(1e-9, 1e-8, color='gray', alpha=0.1)    # Class A
    ax_plt.axhspan(1e-6, 1e-4, color='orange', alpha=0.1)  # Class M
    ax_plt.axhspan(1e-4, 1e-2, color='red', alpha=0.1)     # Class X
    ax_plt.axhline(y=1e-4, color='red', linestyle='--', alpha=0.6) # Порог X
    ax_plt.set_ylim(1e-9, 1e-2)
    ax_plt.set_xlim(0, 50)
    return ax_plt.plot([], [], color='cyan', lw=2)[0] # Та самая переменная

line_plt = setup_plot_style()

# --- 2. Классы объектов ---
class FlareParticle:
    def __init__(self, source, target):
        self.f_class = random.choices(['A', 'B', 'C', 'M', 'X'], weights=[40, 30, 15, 10, 5])[0]
        self.color = COLORS[self.f_class]
        self.speed = {'A':2, 'B':3, 'C':5, 'M':8, 'X':12}[self.f_class]
        self.damage = {'A':0.5, 'B':2, 'C':5, 'M':15, 'X':40}[self.f_class]
        
        # Позиция появления
        if source == 'Sun': self.x, self.y = 80, HEIGHT//2 + random.randint(-150, 150)
        elif source == 'Albedo': self.x, self.y = WIDTH//2, HEIGHT//2 + 80
        else: self.x, self.y = WIDTH//2, HEIGHT//2 + 80 # Earth direct
            
        angle = math.atan2(target[1]-self.y, target[0]-self.x) + random.uniform(-0.1, 0.1)
        self.vx, self.vy = math.cos(angle)*self.speed, math.sin(angle)*self.speed
        self.trail = []
        self.active = True

    def update(self, sim_speed):
        self.trail.append((self.x, self.y))
        if len(self.trail) > 12: self.trail.pop(0)
        self.x += self.vx * sim_speed
        self.y += self.vy * sim_speed
        if not (0 < self.x < WIDTH and 0 < self.y < HEIGHT): self.active = False

    def draw(self, surf):
        for i, pos in enumerate(self.trail):
            alpha = int(255 * (i/len(self.trail)))
            pygame.draw.circle(surf, (*self.color, alpha), (int(pos[0]), int(pos[1])), i//2)
        pygame.draw.circle(surf, (255,255,255), (int(self.x), int(self.y)), 4)

class Satellite:
    def __init__(self, idx, angle):
        self.idx, self.angle, self.orbit = idx, angle, 130
        self.hp, self.temp, self.mode = 100.0, 25.0, "ACTIVE"
        self.x, self.y = 0, 0

    def update(self, center, particles, protection_active, stats, sim_speed):
        self.angle += 0.01 * sim_speed
        self.x = center[0] + self.orbit * math.cos(self.angle)
        self.y = center[1] + self.orbit * math.sin(self.angle)
        
        # Интеллектуальный анализ (Продукт команды)
        danger = False
        if protection_active:
            for p in particles:
                if math.hypot(self.x-p.x, self.y-p.y) < 180 and p.f_class in ['M', 'X']:
                    danger = True
            self.mode = "SHIELD" if danger else "ACTIVE"
        else: self.mode = "ACTIVE"

        for p in particles:
            if p.active and math.hypot(self.x-p.x, self.y-p.y) < 25:
                mult = 0.05 if self.mode == "SHIELD" else 1.2
                dmg = p.damage * mult
                self.hp -= dmg
                self.temp += dmg * 1.5
                if self.mode != "SHIELD": stats['data_lost'] += int(p.damage * 10)
                p.active = False
        
        self.temp = max(25, self.temp - 0.05)
        self.hp = max(0, self.hp)

    def draw(self, surf):
        color = (0, 255, 150) if self.mode == "SHIELD" else (255, 255, 255)
        if self.hp <= 0: color = (80, 80, 80)
         # Корпус спутника (Детализация)
        pygame.draw.rect(surf, color, (self.x-12, self.y-8, 24, 16), 0, 3)
         # Солнечные панели
        panel_ext = 25 if self.mode == "ACTIVE" else 5
        pygame.draw.rect(surf, (50, 80, 200), (self.x-12-panel_ext, self.y-4, panel_ext, 8))
        pygame.draw.rect(surf, (50, 80, 200), (self.x+12, self.y-4, panel_ext, 8))
         # Теги
        txt = font_s.render(f"SAT-0{self.idx} {int(self.hp)}%", True, color)
        surf.blit(txt, (self.x-25, self.y-30))

# --- UI Элементы ---
class Slider:
    def __init__(self, x, y, w, h, label):
        self.rect = pygame.Rect(x, y, w, h)
        self.val = 1.0
        self.label = label

    def draw(self, surf):
        pygame.draw.rect(surf, (60, 60, 80), self.rect, 0, 5)
        circle_x = self.rect.x + int((self.val / 2.0) * self.rect.w)
        pygame.draw.circle(surf, (200, 200, 255), (circle_x, self.rect.centery), 10)
        surf.blit(font_s.render(f"{self.label}: {self.val:.1f}x", True, (255,255,255)), (self.rect.x, self.rect.y-20))

    def update(self, m_pos, m_click, current_speed):
        if self.rect.collidepoint(m_pos) and m_click[0]:
            self.val = round(((m_pos[0] - self.rect.x) / self.rect.w) * 2.0, 1)
            return self.val
        return current_speed

# --- 3. ФУНКЦИЯ ПЕРЕЗАПУСКА (RELOAD) ---
def reset_simulation():
    global particles, sats, stats, flux_data, protection_active, line_plt, shield_lines
    particles = []
    sats = [Satellite(i, i*1.6) for i in range(4)]
    stats = {'data_lost': 0}
    flux_data = []
    protection_active = True
    
    # СТИРАЕМ СТАРЫЕ ЛИНИИ:
    for line in shield_lines:
        line.remove() # Удаляем линию с самого графика
    shield_lines = [] # Очищаем список ссылок
    
    line_plt = setup_plot_style() # Пересоздаем основную голубую линию

# --- 4. Основной запуск ---
earth_pos = (WIDTH // 2 - 100, HEIGHT // 2)
sun_pos = (120, HEIGHT // 2)
speed_slider = Slider(WIDTH - 280, HEIGHT - 60, 220, 15, "Simulation Speed")
sim_speed = 1.0

reset_simulation() 

running = True
while running:
    screen.fill(COLORS['bg'])
    m_pos, m_click = pygame.mouse.get_pos(), pygame.mouse.get_pressed()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT: running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                protection_active = not protection_active
            if event.key == pygame.K_r: 
                reset_simulation()

    sim_speed = speed_slider.update(m_pos, m_click, sim_speed)

    # Отрисовка планет
    # Солнце
    for r in range(5): pygame.draw.circle(screen, (255, 100, 0, 50), sun_pos, 80 + r*10, 1)
    pygame.draw.circle(screen, COLORS['sun'], sun_pos, 80)
    # Земля
    pygame.draw.circle(screen, (40, 60, 150), earth_pos, 65) 
    pygame.draw.circle(screen, COLORS['earth'], earth_pos, 55)
    # Текстура (упрощенно материки)
    pygame.draw.ellipse(screen, (50, 180, 80), (earth_pos[0]-30, earth_pos[1]-20, 40, 25))
    pygame.draw.ellipse(screen, (50, 180, 80), (earth_pos[0]+5, earth_pos[1]+10, 20, 15))

    # Логика частиц
    if random.random() < 0.05 * sim_speed: particles.append(FlareParticle('Sun', earth_pos))
    if random.random() < 0.02 * sim_speed: particles.append(FlareParticle('Albedo', (earth_pos[0]-100, earth_pos[1]-100)))

# 3. Обновление
    for p in particles[:]:
        p.update(sim_speed)
        if not p.active: particles.remove(p)
        else: p.draw(screen)

    for s in sats:
        s.update(earth_pos, particles, protection_active, stats, sim_speed)
        s.draw(screen)

    # UI Панель
    pygame.draw.rect(screen, (15, 20, 35), (WIDTH-320, 0, 320, HEIGHT))
    screen.blit(font_l.render("MISSION CONTROL", True, (0, 255, 200)), (WIDTH-300, 30))
    screen.blit(font_m.render(f"Shield: {'ACTIVE' if protection_active else 'OFF'}", True, (0, 255, 0) if protection_active else (255, 50, 0)), (WIDTH-300, 80))
    screen.blit(font_s.render(f"Data Loss: {stats['data_lost']}", True, (255, 150, 150)), (WIDTH-300, 120))
    screen.blit(font_s.render("Press 'R' to Reset", True, (200, 200, 200)), (WIDTH-300, 150))
    
    y = 200
    for s in sats:
        pygame.draw.rect(screen, (25, 30, 50), (WIDTH-310, y, 300, 90), border_radius=8)
        screen.blit(font_l.render(f"SAT-0{s.idx} | {s.mode}", True, (255,255,255)), (WIDTH-300, y+10))
        # Тепловая шкала
        pygame.draw.rect(screen, (100, 20, 20), (WIDTH-300, y+45, 200, 8))
        pygame.draw.rect(screen, (255, 50, 50), (WIDTH-300, y+45, int(s.hp*2), 8))
        screen.blit(font_s.render(f"Health: {int(s.hp)}%  Temp: {s.temp:.1f}C", True, (200,200,200)), (WIDTH-300, y+60))
        y += 110

    speed_slider.draw(screen)

    # График
    if pygame.time.get_ticks() % 500 < 25:
        # Внутри блока: if pygame.time.get_ticks() % 500 < 25:
     if any(s.mode == "SHIELD" for s in sats):
        # Создаем линию и сразу сохраняем её в наш список
        new_line = ax_plt.axvline(x=len(flux_data)-1, color='red', alpha=0.3, lw=1)
        shield_lines.append(new_line)
        # Сбор данных: берем максимальный поток от всех активных частиц
        current_flux = 1e-9
        if particles:
            # Масштабируем урон под значения потока W/m2
            flux_vals = [p.damage/10000 for p in particles]
            current_flux = max(flux_vals)
        
        flux_data.append(current_flux)
        if len(flux_data) > 50: flux_data.pop(0)
        
        # Обновляем только данные линии, не стирая фон графика
        line_plt.set_data(range(len(flux_data)), flux_data)
        ax_plt.set_xlim(0, 50)
        ax_plt.set_ylim(1e-9, 1e-2)
        
        # Рисуем красную вертикальную линию, если защита сработала
        if any(s.mode == "SHIELD" for s in sats):
            ax_plt.axvline(x=len(flux_data)-1, color='red', alpha=0.3, lw=1)
            
        fig.canvas.draw()
        fig.canvas.flush_events()

    pygame.display.flip()
    clock.tick(60)

pygame.quit()