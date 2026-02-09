// ml/src/static/solar_3d.js

let scene, camera, renderer, panel, sunLight, ambientLight;
let isAutoTilt = false;

// --- 1. INITIALIZE 3D SCENE ---
function init3D() {
    const container = document.getElementById('canvas-container');
    
    if (!container) {
        console.error("Canvas container not found!");
        return;
    }

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xf0f2f5); // Matches app bg

    // Camera
    camera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(0, 3, 6); // Elevated view
    camera.lookAt(0, 0, 0);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true; // Enable shadows
    container.innerHTML = ''; // Clear previous canvas if any
    container.appendChild(renderer.domElement);

    // Ground (to catch shadows)
    const groundGeo = new THREE.PlaneGeometry(20, 20);
    const groundMat = new THREE.MeshStandardMaterial({ color: 0xe0e0e0 });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -1.5;
    ground.receiveShadow = true;
    scene.add(ground);

    // Solar Panel Geometry
    const geometry = new THREE.BoxGeometry(3, 0.1, 2); // Width, Height, Depth
    const material = new THREE.MeshStandardMaterial({ 
        color: 0x1d4e89, // Deep blue
        roughness: 0.3,
        metalness: 0.6,
        emissive: 0x000000,
        emissiveIntensity: 0.0
    });
    panel = new THREE.Mesh(geometry, material);
    panel.castShadow = true;
    scene.add(panel);

    // Stand
    const standGeo = new THREE.CylinderGeometry(0.1, 0.1, 1.5, 32);
    const standMat = new THREE.MeshStandardMaterial({ color: 0x555555 });
    const stand = new THREE.Mesh(standGeo, standMat);
    stand.position.y = -0.75;
    scene.add(stand);

    // Lighting (Sun)
    sunLight = new THREE.DirectionalLight(0xffffff, 1.2);
    sunLight.position.set(5, 10, 5); // Default position
    sunLight.castShadow = true;
    
    // Shadow properties for better quality
    sunLight.shadow.mapSize.width = 1024;
    sunLight.shadow.mapSize.height = 1024;
    
    scene.add(sunLight);
    
    ambientLight = new THREE.AmbientLight(0x404040); // Soft white light
    scene.add(ambientLight);

    // Start Animation Loop
    animate();
    
    // Initialize sun position based on the slider (if it exists)
    const timeSlider = document.getElementById('timeSlider');
    if (timeSlider) {
        updateSunPosition();
    }
}

function animate() {
    requestAnimationFrame(animate);
    
    // Idle animation: slight rotation if not in auto-tilt mode
    // (Optional: remove comments to enable idle spin)
    // if (!isAutoTilt && panel) {
    //    panel.rotation.y += 0.001; 
    // }
    
    renderer.render(scene, camera);
}

// --- 2. 3D INTERACTION LOGIC ---

// Updates Sun position based on Time Slider (6 AM - 6 PM)
function updateSunPosition() {
    const slider = document.getElementById('timeSlider');
    if (!slider) return;

    const hour = parseFloat(slider.value);
    
    // Update Time Badge Text
    const timeDisplay = document.getElementById('timeDisplay');
    if (timeDisplay) {
        const h = Math.floor(hour);
        const m = Math.floor((hour - h) * 60);
        const ampm = h >= 12 ? 'PM' : 'AM';
        const fmtHour = h > 12 && h !== 12 ? h - 12 : (h === 0 || h === 24 ? 12 : h);
        timeDisplay.innerText = `${fmtHour}:${m.toString().padStart(2, '0')} ${ampm}`;
    }

    // Move Light Source (Simple Arc Simulation)
    // Noon (12) is top center. 6 is left, 18 is right.
    const angle = (hour - 12) * (Math.PI / 12); 
    const radius = 8;
    
    sunLight.position.x = Math.sin(angle) * radius; 
    sunLight.position.y = Math.cos(angle) * radius;
    
    // Adjust Light Intensity for Dawn/Dusk
    if (hour < 6.5 || hour > 17.5) {
        ambientLight.intensity = 0.2; // Dim
        sunLight.intensity = 0.5;
    } else {
        ambientLight.intensity = 0.5; // Bright
        sunLight.intensity = 1.2;
    }

    // Auto-Tilt Panel Logic (Dual-Axis Tracking Simulation)
    if (isAutoTilt && panel) {
        panel.rotation.z = -angle; // Rotate panel on Z-axis to face sun
    } else if (panel) {
        panel.rotation.z = 0; // Reset to flat
    }
}

// Toggles the Auto-Tilt mode
function toggleAutoTilt() {
    const toggle = document.getElementById('autoTiltToggle');
    if (toggle) {
        isAutoTilt = toggle.checked;
        updateSunPosition();
    }
}

// Updates Panel Color based on Dust Slider
function update3DFromSliders() {
    if (!panel) return;
    
    const dustInput = document.getElementById('dust_index');
    if (dustInput) {
        const dustLevel = parseFloat(dustInput.value);
        
        // Dust Effect: Interpolate Blue -> Brown
        const cleanColor = new THREE.Color(0x1d4e89);
        const dustColor = new THREE.Color(0x8b7355); 
        panel.material.color.lerpColors(cleanColor, dustColor, dustLevel);
    }
}

// Updates Panel Glow based on Efficiency Prediction
function updateColorByEfficiency(efficiency) {
    if (!panel) return;
    
    // Emissive Glow Logic
    // High Efficiency (>0.8) = Green/Blue Glow
    // Low Efficiency (<0.6) = Red Glow/Heat
    
    if (efficiency < 0.6) {
        panel.material.emissive.setHex(0xff0000); // Red Warning
        panel.material.emissiveIntensity = 0.3;
    } else if (efficiency > 0.8) {
        panel.material.emissive.setHex(0x00ff00); // Green Optimal
        panel.material.emissiveIntensity = 0.1;
    } else {
        panel.material.emissive.setHex(0x000000); // No glow (Normal)
        panel.material.emissiveIntensity = 0;
    }
}

// Handle Window Resize
window.addEventListener('resize', () => {
    if (camera && renderer && document.getElementById('canvas-container')) {
        const container = document.getElementById('canvas-container');
        camera.aspect = container.clientWidth / container.clientHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(container.clientWidth, container.clientHeight);
    }
});