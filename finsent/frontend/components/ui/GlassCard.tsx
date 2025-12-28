'use client';

// =============================================================================
// GLASS CARD COMPONENT
// =============================================================================
// Premium glassmorphism card with gradient border and glow effects
//
// Location: frontend/components/ui/GlassCard.tsx
// =============================================================================

import React from 'react';

interface GlassCardProps {
    children: React.ReactNode;
    /** Additional className */
    className?: string;
    /** Enable gradient border on hover */
    gradientBorder?: boolean;
    /** Enable glow effect on hover */
    glow?: boolean;
    /** Glow color variant */
    glowColor?: 'terra' | 'navy' | 'success' | 'warning';
    /** Enable hover lift effect */
    hover?: boolean;
    /** Card padding */
    padding?: 'sm' | 'md' | 'lg' | 'xl';
    /** Background blur intensity */
    blur?: 'sm' | 'md' | 'lg';
    /** On click handler */
    onClick?: () => void;
}

const paddingClasses = {
    sm: 'p-3',
    md: 'p-4',
    lg: 'p-6',
    xl: 'p-8',
};

const blurClasses = {
    sm: 'backdrop-blur-sm',
    md: 'backdrop-blur-md',
    lg: 'backdrop-blur-lg',
};

const glowColors = {
    terra: 'hover:shadow-[0_0_40px_rgba(149,76,46,0.15)]',
    navy: 'hover:shadow-[0_0_40px_rgba(37,77,112,0.15)]',
    success: 'hover:shadow-[0_0_40px_rgba(45,122,79,0.15)]',
    warning: 'hover:shadow-[0_0_40px_rgba(154,107,40,0.15)]',
};

export const GlassCard: React.FC<GlassCardProps> = ({
    children,
    className = '',
    gradientBorder = false,
    glow = false,
    glowColor = 'terra',
    hover = true,
    padding = 'lg',
    blur = 'md',
    onClick,
}) => {
    const baseClasses = `
    relative
    bg-white/80
    ${blurClasses[blur]}
    rounded-xl
    border border-white/20
    shadow-lg
    ${paddingClasses[padding]}
    transition-all duration-300 ease-out
  `;

    const hoverClasses = hover ? `
    hover:-translate-y-1
    hover:shadow-xl
  ` : '';

    const glowClasses = glow ? glowColors[glowColor] : '';

    const cursorClass = onClick ? 'cursor-pointer' : '';

    if (gradientBorder) {
        return (
            <div
                className={`relative group ${cursorClass} ${className}`}
                onClick={onClick}
            >
                {/* Gradient border background */}
                <div className="absolute -inset-[1px] bg-gradient-to-r from-terra-400 via-navy-400 to-terra-400 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300 blur-[1px]" />

                {/* Inner card */}
                <div className={`
          relative
          bg-white
          ${blurClasses[blur]}
          rounded-xl
          ${paddingClasses[padding]}
          transition-all duration-300 ease-out
          ${hover ? 'group-hover:-translate-y-1 group-hover:shadow-xl' : ''}
          ${glowClasses}
        `}>
                    {children}
                </div>
            </div>
        );
    }

    return (
        <div
            className={`${baseClasses} ${hoverClasses} ${glowClasses} ${cursorClass} ${className}`}
            onClick={onClick}
        >
            {children}
        </div>
    );
};

// =============================================================================
// GLASS PANEL - Alternate style for sections
// =============================================================================

interface GlassPanelProps {
    children: React.ReactNode;
    className?: string;
    dark?: boolean;
    padding?: 'sm' | 'md' | 'lg' | 'xl';
}

export const GlassPanel: React.FC<GlassPanelProps> = ({
    children,
    className = '',
    dark = false,
    padding = 'lg',
}) => {
    const bgClass = dark
        ? 'bg-navy-900/80 text-white'
        : 'bg-white/60';

    return (
        <div className={`
      ${bgClass}
      backdrop-blur-lg
      rounded-2xl
      border border-white/10
      shadow-2xl
      ${paddingClasses[padding]}
      ${className}
    `}>
            {children}
        </div>
    );
};

export default GlassCard;
