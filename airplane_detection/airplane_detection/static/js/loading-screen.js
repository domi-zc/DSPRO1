document.addEventListener("DOMContentLoaded", function() {
    initLoadingScreenTrigger();
});

function initLoadingScreenTrigger() {
    const form = document.getElementById("ap-image-form");
    const loadingScreen = document.getElementById("ap-loading-screen");

    if (!form || !loadingScreen) return;

    // Startet erst, wenn das Formular abgeschickt wird
    form.addEventListener("submit", function(e) {

        // Animation starten
        startGsapAnimation(loadingScreen);

        // Screen sichtbar machen
        loadingScreen.classList.add("active");

        // Scrollen verbieten
        document.documentElement.style.overflow = "hidden";
    });
}

function startGsapAnimation(loadingScreen) {
    if (typeof gsap === 'undefined') {
        console.error("GSAP library not loaded!");
        return;
    }

    const svg = loadingScreen.querySelector(".ap-loading-svg");
    const mainContainer = loadingScreen.querySelector(".ap-loading-container");
    const planeGroup = loadingScreen.querySelector(".ap-plane-group");
    const plane = loadingScreen.querySelector(".ap-plane");
    const dotTemplate = loadingScreen.querySelector(".ap-dot");

    // SVG sichtbar machen
    gsap.set(svg, {
        visibility: 'visible'
    });

    // --- Konfiguration ---
    const centerX = 400;
    const centerY = 300;
    const radius = 80;
    const numDots = 30;
    const loopDuration = 2.5;

    // Container leeren
    mainContainer.innerHTML = "";

    // Punkte erstellen
    for (let i = 0; i < numDots; i++) {
        const angle = (360 / numDots) * i;
        const x = centerX + Math.cos((angle * Math.PI) / 180) * radius;
        const y = centerY + Math.sin((angle * Math.PI) / 180) * radius;

        const newDot = dotTemplate.cloneNode(true);
        
        newDot.setAttribute("cx", x);
        newDot.setAttribute("cy", y);
        newDot.setAttribute("r", 0);
        mainContainer.appendChild(newDot);

        const dotDelay = (i / numDots) * loopDuration + 0.05;

        gsap.to(newDot, {
            keyframes: [{
                    attr: {
                        r: 5,
                        opacity: 1
                    },
                    duration: loopDuration * 0.1,
                    ease: "power2.out"
                },
                {
                    attr: {
                        r: 0,
                        opacity: 0.5
                    },
                    duration: loopDuration * 0.9,
                    ease: "power1.in"
                }
            ],
            repeat: -1,
            delay: dotDelay,
            immediateRender: false
        });
    }

    // Flugzeug Position
    gsap.set(planeGroup, {
        x: centerX,
        y: centerY
    });

    gsap.set(plane, {
        x: radius,
        y: 0,
        xPercent: -50,
        yPercent: -50,
        rotation: 90,
        transformOrigin: "center center"
    });

    // Flugzeug Rotation
    gsap.to(planeGroup, {
        rotation: 360,
        duration: loopDuration,
        repeat: -1,
        ease: "none"
    });
}