'use client';

// =============================================================================
// MESH GRADIENT BACKGROUND
// =============================================================================
// Animated mesh gradient component using Tailwind CSS
// =============================================================================

import React from 'react';

interface MeshGradientProps {
  className?: string;
  intensity?: number;
  animated?: boolean;
}

export const MeshGradient: React.FC<MeshGradientProps> = ({
  className = '',
  intensity = 1,
  animated = true,
}) => {
  return (
    <div className={`absolute inset-0 overflow-hidden bg-gradient-to-br from-cream-50 to-cream-100 ${className}`}>
      {/* Base gradient layer */}
      <div
        className="absolute inset-0"
        style={{
          opacity: intensity,
          background: `
            radial-gradient(ellipse 80% 50% at 50% 120%, rgba(249, 115, 22, 0.15) 0%, transparent 50%),
            radial-gradient(ellipse 60% 40% at 20% 100%, rgba(236, 72, 153, 0.12) 0%, transparent 50%),
            radial-gradient(ellipse 50% 50% at 80% 90%, rgba(124, 58, 237, 0.1) 0%, transparent 50%),
            radial-gradient(ellipse 80% 60% at 50% -20%, rgba(59, 130, 246, 0.08) 0%, transparent 50%)
          `
        }}
      />

      {/* Animated orbs */}
      <div
        className={`absolute w-[60%] h-[60%] -bottom-[20%] -left-[10%] rounded-full blur-[80px] ${animated ? 'animate-pulse' : ''}`}
        style={{
          opacity: intensity * 0.8,
          background: 'linear-gradient(135deg, rgba(249, 115, 22, 0.4) 0%, rgba(236, 72, 153, 0.3) 100%)'
        }}
      />
      <div
        className={`absolute w-[50%] h-[50%] -bottom-[15%] -right-[5%] rounded-full blur-[80px] ${animated ? 'animate-pulse' : ''}`}
        style={{
          opacity: intensity * 0.7,
          background: 'linear-gradient(135deg, rgba(236, 72, 153, 0.35) 0%, rgba(124, 58, 237, 0.25) 100%)',
          animationDelay: '1s'
        }}
      />
      <div
        className={`absolute w-[40%] h-[40%] top-[10%] right-[10%] rounded-full blur-[80px] ${animated ? 'animate-pulse' : ''}`}
        style={{
          opacity: intensity * 0.6,
          background: 'linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(124, 58, 237, 0.15) 100%)',
          animationDelay: '2s'
        }}
      />
      <div
        className={`absolute w-[35%] h-[35%] bottom-[20%] left-[30%] rounded-full blur-[80px] ${animated ? 'animate-pulse' : ''}`}
        style={{
          opacity: intensity * 0.5,
          background: 'linear-gradient(135deg, rgba(249, 115, 22, 0.25) 0%, rgba(236, 72, 153, 0.2) 100%)',
          animationDelay: '3s'
        }}
      />

      {/* Subtle noise texture */}
      <div
        className="absolute inset-0 opacity-[0.03] pointer-events-none"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E")`
        }}
      />
    </div>
  );
};

export default MeshGradient;
