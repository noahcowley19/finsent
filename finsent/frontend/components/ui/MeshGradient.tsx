'use client';

// =============================================================================
// MESH GRADIENT BACKGROUND
// =============================================================================
// Animated mesh gradient component inspired by Lovable.com
// Creates a dynamic, colorful background with smooth gradient animations
//
// Location: frontend/components/ui/MeshGradient.tsx
// =============================================================================

import React from 'react';

interface MeshGradientProps {
  /** Additional CSS classes */
  className?: string;
  /** Intensity of the gradient (0-1) */
  intensity?: number;
  /** Whether to animate the gradient */
  animated?: boolean;
}

export const MeshGradient: React.FC<MeshGradientProps> = ({
  className = '',
  intensity = 1,
  animated = true,
}) => {
  return (
    <div className={`mesh-gradient-container ${className}`}>
      {/* Base gradient layer */}
      <div 
        className="mesh-gradient-base"
        style={{ opacity: intensity }}
      />
      
      {/* Animated orbs */}
      <div 
        className={`mesh-gradient-orb mesh-gradient-orb-1 ${animated ? 'mesh-gradient-animate-1' : ''}`}
        style={{ opacity: intensity * 0.8 }}
      />
      <div 
        className={`mesh-gradient-orb mesh-gradient-orb-2 ${animated ? 'mesh-gradient-animate-2' : ''}`}
        style={{ opacity: intensity * 0.7 }}
      />
      <div 
        className={`mesh-gradient-orb mesh-gradient-orb-3 ${animated ? 'mesh-gradient-animate-3' : ''}`}
        style={{ opacity: intensity * 0.6 }}
      />
      <div 
        className={`mesh-gradient-orb mesh-gradient-orb-4 ${animated ? 'mesh-gradient-animate-4' : ''}`}
        style={{ opacity: intensity * 0.5 }}
      />

      {/* Noise texture overlay for depth */}
      <div className="mesh-gradient-noise" />

      <style jsx>{`
        .mesh-gradient-container {
          position: absolute;
          inset: 0;
          overflow: hidden;
          background: linear-gradient(135deg, #FAF7F2 0%, #f5ede0 100%);
        }

        .mesh-gradient-base {
          position: absolute;
          inset: 0;
          background: 
            radial-gradient(ellipse 80% 50% at 50% 120%, rgba(249, 115, 22, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse 60% 40% at 20% 100%, rgba(236, 72, 153, 0.12) 0%, transparent 50%),
            radial-gradient(ellipse 50% 50% at 80% 90%, rgba(124, 58, 237, 0.1) 0%, transparent 50%),
            radial-gradient(ellipse 80% 60% at 50% -20%, rgba(59, 130, 246, 0.08) 0%, transparent 50%);
        }

        .mesh-gradient-orb {
          position: absolute;
          border-radius: 50%;
          filter: blur(80px);
          mix-blend-mode: normal;
        }

        .mesh-gradient-orb-1 {
          width: 60%;
          height: 60%;
          bottom: -20%;
          left: -10%;
          background: linear-gradient(135deg, rgba(249, 115, 22, 0.4) 0%, rgba(236, 72, 153, 0.3) 100%);
        }

        .mesh-gradient-orb-2 {
          width: 50%;
          height: 50%;
          bottom: -15%;
          right: -5%;
          background: linear-gradient(135deg, rgba(236, 72, 153, 0.35) 0%, rgba(124, 58, 237, 0.25) 100%);
        }

        .mesh-gradient-orb-3 {
          width: 40%;
          height: 40%;
          top: 10%;
          right: 10%;
          background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(124, 58, 237, 0.15) 100%);
        }

        .mesh-gradient-orb-4 {
          width: 35%;
          height: 35%;
          bottom: 20%;
          left: 30%;
          background: linear-gradient(135deg, rgba(249, 115, 22, 0.25) 0%, rgba(236, 72, 153, 0.2) 100%);
        }

        .mesh-gradient-animate-1 {
          animation: meshFloat1 20s ease-in-out infinite;
        }

        .mesh-gradient-animate-2 {
          animation: meshFloat2 25s ease-in-out infinite;
        }

        .mesh-gradient-animate-3 {
          animation: meshFloat3 18s ease-in-out infinite;
        }

        .mesh-gradient-animate-4 {
          animation: meshFloat4 22s ease-in-out infinite;
        }

        @keyframes meshFloat1 {
          0%, 100% {
            transform: translate(0, 0) scale(1);
          }
          25% {
            transform: translate(5%, -5%) scale(1.05);
          }
          50% {
            transform: translate(-3%, -8%) scale(0.98);
          }
          75% {
            transform: translate(8%, 3%) scale(1.02);
          }
        }

        @keyframes meshFloat2 {
          0%, 100% {
            transform: translate(0, 0) scale(1);
          }
          33% {
            transform: translate(-8%, 5%) scale(1.08);
          }
          66% {
            transform: translate(5%, -3%) scale(0.95);
          }
        }

        @keyframes meshFloat3 {
          0%, 100% {
            transform: translate(0, 0) scale(1);
          }
          50% {
            transform: translate(-10%, 10%) scale(1.1);
          }
        }

        @keyframes meshFloat4 {
          0%, 100% {
            transform: translate(0, 0) scale(1) rotate(0deg);
          }
          25% {
            transform: translate(15%, -5%) scale(1.05) rotate(5deg);
          }
          50% {
            transform: translate(5%, 10%) scale(0.95) rotate(-3deg);
          }
          75% {
            transform: translate(-10%, 5%) scale(1.08) rotate(2deg);
          }
        }

        .mesh-gradient-noise {
          position: absolute;
          inset: 0;
          opacity: 0.03;
          background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
          pointer-events: none;
        }
      `}</style>
    </div>
  );
};

export default MeshGradient;
