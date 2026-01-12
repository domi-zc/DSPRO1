/**
 * -----------------------------------------------------------------------------
 * AUTOMATISCHE SLIDER-ANIMATION FÜR VIDEO-AUFNAHMEN
 * -----------------------------------------------------------------------------
 * Dieses Skript automatisiert den "Image Comparison Slider", um eine
 * Bildschirmaufnahme zu ermöglichen.
 *
 * ANLEITUNG:
 * 1. Öffne die Entwicklerkonsole
 * 2. Tippe "demo" ein und drücke Enter
 *
 */

Object.defineProperty(window, 'demo', {
  get: () => {
    // in ms
    const startDelay = 3000;     // Wartezeit am Anfang
    const slideEffectDuration = 10000;   // Dauer der Slider-Bewegung
    const pauseDuration = 1000;               // Pause für slide Effekt Links Rechts am Rand und 
    const endFlashColor = 1000;       // Dauer des Black-Flash am Ende

    // Hintergrund-Farben
    manageContainerEffects(startDelay, slideEffectDuration, pauseDuration, endFlashColor);

    // Slider-Animation
    setTimeout(() => {
      runSliderAnimation(slideEffectDuration, pauseDuration);
    }, startDelay);

    return "Animation startet in 3 Sekunden...";
  },
  configurable: true
});

// Hintergrundfarbe steuern 
function manageContainerEffects(delay, slideEffectDuration, pauseDuration, endFlashColor) {
  const container = document.querySelector(".ap-output-image-comparison");
  const backgroundColor = "#000"
  if (!container) return;

  // Start -> Schwarz für die 3 Sekunden Countdown bis Slider beginnt zu bewegen
  container.style.backgroundColor = backgroundColor;

  // Nach Countdown reset auf standardfarbe -> Animation läuft
  setTimeout(() => {
    container.style.backgroundColor = "";
  }, delay);

  // Nachdem Animation beendet wurde für eine Sekunde Pause und dann zeigt es wieder schwarz an
  setTimeout(() => {
    container.style.backgroundColor = backgroundColor;

    // Ende -> Nach kurzem Anzeigen verschwindet wieder und Funktion ist beendet
    setTimeout(() => {
      container.style.backgroundColor = "";
    }, endFlashColor);

  }, delay + slideEffectDuration + pauseDuration);
}

// Border Radius von Bildern auf 0 setzen
function toggleImageStyles(active) {
  const images = document.querySelectorAll("#ap-compare-image img");
  images.forEach(img => {
    img.style.borderRadius = active ? "0" : "";
  });
}

// Slider Bewegung
function runSliderAnimation(totalDuration, pauseDuration) {
  const el = document.getElementById("ap-compare-image");
  const start = performance.now();

  toggleImageStyles(true); // Border-Radius auf 0 setzen

  const startPos = 50;
  const min = 2.3;
  const max = 97.7;

  // Distanzen: 47.7 + 95.4 + 47.7 = 190.8
  const totalDist = 2 * (max - min);

  const moveTime = totalDuration - (pauseDuration * 2);
  const f = moveTime / totalDist; 

  const t1 = (max - startPos) * f;
  const t2 = t1 + pauseDuration;
  const t3 = t2 + (max - min) * f;
  const t4 = t3 + pauseDuration;

  function loop(now) {
    const t = now - start;
    let p = startPos;

    if (t < t1) p = startPos + t / f;                   
    else if (t < t2) p = max;                    
    else if (t < t3) p = max - (t - t2) / f;     
    else if (t < t4) p = min;                     
    else if (t < totalDuration) p = min + (t - t4) / f; 
    else {
      el.style.setProperty("--pos", "50%");
      setTimeout(() => {
        toggleImageStyles(false); // Border-Radius zurücksetzen
      }, pauseDuration);
      return;
    }

    el.style.setProperty("--pos", p + "%");
    requestAnimationFrame(loop);
  }
  
  requestAnimationFrame(loop);
}