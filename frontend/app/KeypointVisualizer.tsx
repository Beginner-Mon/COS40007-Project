'use client';
import { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';

interface FrameData {
    frame: number;
    keypoints: [number, number, number][];
}

interface KeypointData {
    joints: string[];
    fps: number;
    frames: FrameData[];
}

export default function KeypointVisualizer() {
    const containerRef = useRef<HTMLDivElement>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const controlsRef = useRef<OrbitControls | null>(null);
    const pointsRef = useRef<THREE.Points[]>([]);
    const [data, setData] = useState<KeypointData | null>(null);
    const [currentFrame, setCurrentFrame] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [loading, setLoading] = useState(true);
    const animationRef = useRef<number | null>(null);
    const lastFrameTimeRef = useRef<number>(0);

    // Load data from JSON
    useEffect(() => {
        fetch('/keypoints_data.json')
            .then(res => res.json())
            .then(data => {
                setData(data);
                setLoading(false);
            })
            .catch(err => {
                console.error('Failed to load keypoint data:', err);
                setLoading(false);
            });
    }, []);

    // Initialize Three.js scene
    useEffect(() => {
        if (!containerRef.current || !data) return;

        // Scene setup
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);
        sceneRef.current = scene;

        // Camera setup
        const camera = new THREE.PerspectiveCamera(
            75,
            containerRef.current.clientWidth / containerRef.current.clientHeight,
            0.1,
            1000
        );
        camera.position.z = 2;
        camera.position.y = 0.5;
        cameraRef.current = camera;

        // Renderer setup
        const renderer = new THREE.WebGLRenderer({ antialias: true });
        renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        containerRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // OrbitControls setup
        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.autoRotate = false;
        controls.autoRotateSpeed = 2;
        controls.enableZoom = true;
        controls.enablePan = true;
        controls.enableRotate = true;
        controlsRef.current = controls;

        // Create point materials with different colors for visibility
        const material = new THREE.PointsMaterial({
            color: 0x00ff00,
            size: 0.05,
            sizeAttenuation: true,
        });

        // Create points for each joint
        const points: THREE.Points[] = [];
        for (let i = 0; i < data.joints.length; i++) {
            const geometry = new THREE.BufferGeometry();
            const position = new Float32Array([0, 0, 0]);
            geometry.setAttribute('position', new THREE.BufferAttribute(position, 3));

            const pt = new THREE.Points(geometry, material);
            scene.add(pt);
            points.push(pt);
        }
        pointsRef.current = points;

        // Add light
        const light = new THREE.PointLight(0xffffff, 1);
        light.position.set(5, 5, 5);
        scene.add(light);

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        scene.add(ambientLight);

        // Handle window resize
        const handleResize = () => {
            if (!containerRef.current) return;
            const width = containerRef.current.clientWidth;
            const height = containerRef.current.clientHeight;
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
            renderer.setSize(width, height);
        };

        window.addEventListener('resize', handleResize);

        // Cleanup
        return () => {
            window.removeEventListener('resize', handleResize);
            controls.dispose();
            if (containerRef.current && renderer.domElement.parentNode === containerRef.current) {
                containerRef.current.removeChild(renderer.domElement);
            }
            geometry.dispose();
            material.dispose();
            renderer.dispose();
        };
    }, [data]);

    // Animation loop
    useEffect(() => {
        if (!data || !rendererRef.current || !sceneRef.current || !cameraRef.current) return;

        const frameInterval = 1000 / data.fps; // milliseconds per frame

        const animate = (currentTime: number) => {
            if (isPlaying) {
                if (lastFrameTimeRef.current === 0) {
                    lastFrameTimeRef.current = currentTime;
                }

                const elapsed = currentTime - lastFrameTimeRef.current;

                if (elapsed >= frameInterval) {
                    setCurrentFrame(prev => {
                        const next = prev + 1;
                        return next >= data.frames.length ? 0 : next;
                    });
                    lastFrameTimeRef.current = currentTime;
                }
            }

            // Update point positions for current frame
            if (currentFrame < data.frames.length) {
                const frameData = data.frames[currentFrame];
                frameData.keypoints.forEach((keypoint, idx) => {
                    if (idx < pointsRef.current.length) {
                        const positions = pointsRef.current[idx].geometry.attributes.position.array as Float32Array;
                        positions[0] = keypoint[0];
                        positions[1] = keypoint[1];
                        positions[2] = keypoint[2];
                        pointsRef.current[idx].geometry.attributes.position.needsUpdate = true;
                    }
                });
            }

            // Render
            rendererRef.current!.render(sceneRef.current!, cameraRef.current!);
            if (controlsRef.current) {
                controlsRef.current.update();
            }
            animationRef.current = requestAnimationFrame(animate);
        };

        animationRef.current = requestAnimationFrame(animate);

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
        };
    }, [data, isPlaying, currentFrame]);

    if (loading) {
        return <div className="flex items-center justify-center h-screen text-white">Loading keypoint data...</div>;
    }

    if (!data) {
        return <div className="flex items-center justify-center h-screen text-white">Failed to load keypoint data</div>;
    }

    const totalFrames = data.frames.length;
    const currentTime = (currentFrame / data.fps).toFixed(2);

    const setCameraView = (position: [number, number, number]) => {
        if (cameraRef.current && controlsRef.current) {
            cameraRef.current.position.set(...position);
            controlsRef.current.target.set(0, 0, 0);
            controlsRef.current.update();
        }
    };

    return (
        <div className="flex flex-col h-screen bg-black text-white">
            <div className="flex-1 overflow-hidden" ref={containerRef} />

            <div className="bg-gray-900 p-6 border-t border-gray-700">
                <div className="flex items-center gap-6 mb-4">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium"
                    >
                        {isPlaying ? 'Pause' : 'Play'}
                    </button>

                    <button
                        onClick={() => setCurrentFrame(0)}
                        className="px-6 py-2 bg-gray-700 hover:bg-gray-600 rounded font-medium"
                    >
                        Reset
                    </button>

                    <div className="flex-1 flex items-center gap-4">
                        <input
                            type="range"
                            min="0"
                            max={totalFrames - 1}
                            value={currentFrame}
                            onChange={e => {
                                setCurrentFrame(Number(e.target.value));
                                setIsPlaying(false);
                            }}
                            className="flex-1 cursor-pointer"
                        />
                    </div>

                    <div className="text-sm text-gray-400 font-mono">
                        Frame: {currentFrame} / {totalFrames - 1} | Time: {currentTime}s
                    </div>
                </div>

                <div className="flex items-center gap-3 mb-4">
                    <span className="text-sm text-gray-400">Camera Views:</span>
                    <button
                        onClick={() => setCameraView([2, 0.5, 0])}
                        className="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm"
                    >
                        Front
                    </button>
                    <button
                        onClick={() => setCameraView([-2, 0.5, 0])}
                        className="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm"
                    >
                        Back
                    </button>
                    <button
                        onClick={() => setCameraView([0, 0.5, 2])}
                        className="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm"
                    >
                        Right
                    </button>
                    <button
                        onClick={() => setCameraView([0, 0.5, -2])}
                        className="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm"
                    >
                        Left
                    </button>
                    <button
                        onClick={() => setCameraView([1.5, 2, 1.5])}
                        className="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm"
                    >
                        Top-Front
                    </button>
                    <button
                        onClick={() => setCameraView([1.5, 1.5, 1.5])}
                        className="px-3 py-1 bg-purple-600 hover:bg-purple-700 rounded text-sm"
                    >
                        Isometric
                    </button>
                </div>

                <div className="text-xs text-gray-500">
                    <p>Joints: {data.joints.length} | FPS: {data.fps} | Total Frames: {totalFrames}</p>
                    <p className="mt-2">
                        {data.joints.map((joint, idx) => (
                            <span key={idx} className="mr-4">
                                {idx + 1}. {joint}
                            </span>
                        ))}
                    </p>
                </div>
            </div>
        </div>
    );
}
