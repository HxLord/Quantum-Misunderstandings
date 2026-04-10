const viewer = new Cesium.Viewer("cesiumContainer", {
  shouldAnimate: true,
  selectionIndicator: false,
  baseLayerPicker: false,
});

// Настройки для визуализации канала связи
const earthRadius = Cesium.Ellipsoid.WGS84.maximumRadius;
const satelliteHeight = 35786000; // Геостационарная
const satPosition = Cesium.Cartesian3.fromDegrees(0, 0, satelliteHeight);
const groundStationPosition = Cesium.Cartesian3.fromDegrees(0, 0, 0);

// --- 1. Визуализация объектов ---

// Наземная станция
viewer.entities.add({
  name: "Ground Station",
  position: groundStationPosition,
  point: { pixelSize: 10, color: Cesium.Color.BLACK },
  label: { text: "Mission Control", font: "10pt sans-serif", pixelOffset: new Cesium.Cartesian2(0, 15) }
});

// Спутник
const satellite = viewer.entities.add({
  name: "Solar Sentinel",
  position: satPosition,
  point: {
    pixelSize: 15,
    color: Cesium.Color.CYAN,
    outlineColor: Cesium.Color.WHITE,
    outlineWidth: 2,
  },
  label: {
    text: "Connecting to NASA...",
    font: "12pt monospace",
    style: Cesium.LabelStyle.FILL_AND_OUTLINE,
    outlineWidth: 2,
    verticalOrigin: Cesium.VerticalOrigin.BOTTOM,
    pixelOffset: new Cesium.Cartesian2(0, -20),
  },
});

// Линия связи (по умолчанию зеленая)
const commsLine = viewer.entities.add({
  name: "Comms Link",
  polyline: {
    positions: [satPosition, groundStationPosition],
    width: 3,
    material: new Cesium.PolylineDashMaterialProperty({
      color: Cesium.Color.GREEN,
      dashLength: 16,
    }),
  },
});

// --- 2. Логика API и Аналитики (Ваш проект) ---

let currentFlareClass = "Unknown";
let isSafeMode = false;

// Функция запроса к NASA DONKI
async function fetchNASAData() {
  console.log("Polling NASA DONKI API for Sun activity...");
  
  // Ключ DEMO_KEY имеет лимиты. Если они кончатся, спутник перейдет в Error.
  const apiKey = "DEMO_KEY"; 
  const today = new Date().toISOString().split('T')[0];
  const url = `https://api.nasa.gov/DONKI/FLR?startDate=${today}&endDate=${today}&api_key=${apiKey}`;

  try {
    const response = await fetch(url);
    const data = await response.json();

    if (data && data.length > 0) {
      currentFlareClass = data[data.length - 1].classType; // Берем последнюю
    } else {
      currentFlareClass = "A0.0"; // Тихое Солнце
    }
    analyzeSolarThreat(currentFlareClass);
  } catch (error) {
    console.error("NASA API API Error:", error);
    satellite.label.text = "NASA Link Error";
    currentFlareClass = "Error";
    analyzeSolarThreat("Error");
  }
}

// --- 3. Логика реакции и сброса данных ---

function analyzeSolarThreat(type) {
  let satColor, satAction, lineMaterial, lineStatus, mode;

  // Текстовое описание класса вспышки для лога
  let threatLevel = type.charAt(0);

  switch (threatLevel) {
    case 'X': // ЭКСТРЕМАЛЬНАЯ (Вариант Выживания)
      satColor = Cesium.Color.RED;
      satAction = "CRITICAL: X-CLASS FLARE!\nAction: EMERGENCY SAFE MODE.";
      
      // Полный обрыв связи
      lineStatus = "SIGNAL LOST (Ionospehere Interference)";
      lineMaterial = new Cesium.PolylineDashMaterialProperty({
          color: Cesium.Color.RED,
          dashLength: 0.1, // "Мертвая" линия
      });
      isSafeMode = true;
      break;

    case 'M': // СИЛЬНАЯ (Вариант Alert)
      satColor = Cesium.Color.ORANGE;
      satAction = "ALERT: M-CLASS FLARE.\nAction: Low-gain antenna active.";
      
      // Связь с перебоями
      lineStatus = "Link Unstable";
      lineMaterial = new Cesium.PolylineDashMaterialProperty({
          color: Cesium.Color.ORANGE,
          dashLength: 4, // Частый пульс
      });
      isSafeMode = false;
      break;

    case 'C': // УМЕРЕННАЯ
      satColor = Cesium.Color.YELLOW;
      satAction = "WARNING: C-CLASS FLARE.\nAction: Logging data.";
      lineStatus = "Nominal (Reduced Rate)";
      lineMaterial = new Cesium.PolylineDashMaterialProperty({
          color: Cesium.Color.YELLOW,
          dashLength: 16,
      });
      isSafeMode = false;
      break;

    case 'A': case 'B': // ТИХАЯ
      satColor = Cesium.Color.GREEN;
      satAction = "STATUS: OK. Quiet Sun.";
      lineStatus = "Nominal (High Rate)";
      lineMaterial = new Cesium.PolylineDashMaterialProperty({
          color: Cesium.Color.GREEN,
          dashLength: 16,
      });
      isSafeMode = false;
      break;
      
    default: // Error / Initialization
      satColor = Cesium.Color.GRAY;
      satAction = "Waiting for data...";
      lineStatus = "Standby";
      lineMaterial = new Cesium.PolylineDashMaterialProperty({ color: Cesium.Color.GRAY });
  }

  // Применяем изменения к Cesium Entities
  satellite.point.color = satColor;
  satellite.label.text = `Class: ${type}\n${satAction}`;
  
  commsLine.polyline.material = lineMaterial;
  commsLine.name = `Comms: ${lineStatus}`;

  console.log(`[Satellite] Detected: ${type}. Mode: ${satAction.split('\n')[1]}`);
}

// --- 4. Запуск симуляции ---

// Проверяем солнце каждые 20 секунд (чтобы не превысить лимит DEMO_KEY)
setInterval(fetchNASAData, 20000);
fetchNASAData(); // Первый запуск

// Камера на спутник
viewer.trackedEntity = satellite;
viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(0, -30, 80000000)
});
