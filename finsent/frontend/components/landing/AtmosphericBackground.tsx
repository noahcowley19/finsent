'use client';

import React, { ReactNode } from 'react';

// =============================================================================
// ATMOSPHERIC BACKGROUND COMPONENT
// =============================================================================
// Creates the warm cream background with subtle light leak effects in corners.
// Muted coral top-left, soft lavender bottom-right.
// =============================================================================

export interface AtmosphericBackgroundProps {
    children: ReactNode;
    className?: string;
    variant?: 'default' | 'subtle' | 'intense';
}

export const AtmosphericBackground: React.FC<AtmosphericBackgroundProps> = ({
    children,
    className = '',
    variant = 'default',
}) => {
    const intensityMap = {
        subtle: { coral: 0.15, lavender: 0.1 },
        default: { coral: 0.25, lavender: 0.2 },
        intense: { coral: 0.35, lavender: 0.3 },
    };

    const intensity = intensityMap[variant];

    return (
        <div className={`relative overflow-hidden ${className}`}>
            {/* Base cream background */}
            <div
                className="absolute inset-0"
                style={{ backgroundColor: '#FDFCF9' }}
            />

            {/* Top-left coral light leak */}
            <div
                className="absolute pointer-events-none animate-[lightPulse_8s_ease-in-out_infinite]"
                style={{
                    top: '-15%',
                    left: '-10%',
                    width: '55%',
                    height: '55%',
                    background: `radial-gradient(ellipse at center, rgba(255, 180, 171, ${intensity.coral}) 0%, transparent 70%)`,
                }}
            />

            {/* Bottom-right lavender light leak */}
            <div
                className="absolute pointer-events-none animate-[lightPulse_8s_ease-in-out_infinite]"
                style={{
                    bottom: '-15%',
                    right: '-10%',
                    width: '55%',
                    height: '55%',
                    background: `radial-gradient(ellipse at center, rgba(196, 181, 253, ${intensity.lavender}) 0%, transparent 70%)`,
                    animationDelay: '4s',
                }}
            />

            {/* Optional top-right peach accent */}
            <div
                className="absolute pointer-events-none opacity-60"
                style={{
                    top: '10%',
                    right: '5%',
                    width: '30%',
                    height: '30%',
                    background: `radial-gradient(ellipse at center, rgba(255, 218, 185, 0.12) 0%, transparent 70%)`,
                }}
            />

            {/* Content */}
            <div className="relative z-10">
                {children}
            </div>
        </div>
    );
};

export default AtmosphericBackground;
