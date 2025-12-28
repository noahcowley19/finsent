'use client';

import React, { forwardRef, HTMLAttributes, ReactNode } from 'react';

// =============================================================================
// TYPES
// =============================================================================

export type GlassVariant = 'light' | 'dark' | 'subtle';
export type GlassPadding = 'none' | 'sm' | 'md' | 'lg';

export interface GlassCardProps extends HTMLAttributes<HTMLDivElement> {
    /** Glass variant */
    variant?: GlassVariant;
    /** Padding */
    padding?: GlassPadding;
    /** Enable hover effects */
    hover?: boolean;
    /** Border */
    border?: boolean;
    /** Content */
    children: ReactNode;
}

// =============================================================================
// STYLES
// =============================================================================

const variantStyles: Record<GlassVariant, string> = {
    light: 'bg-white/80 backdrop-blur-xl',
    dark: 'bg-ink-900/80 backdrop-blur-xl',
    subtle: 'bg-white/60 backdrop-blur-lg',
};

const borderStyles: Record<GlassVariant, string> = {
    light: 'border border-white/20',
    dark: 'border border-white/10',
    subtle: 'border border-ink-200/30',
};

const paddingStyles: Record<GlassPadding, string> = {
    none: 'p-0',
    sm: 'p-4',
    md: 'p-6',
    lg: 'p-8',
};

const hoverStyles: Record<GlassVariant, string> = {
    light: 'hover:-translate-y-0.5 hover:shadow-glass-lg hover:border-white/30',
    dark: 'hover:-translate-y-0.5 hover:shadow-glass-lg hover:border-white/15',
    subtle: 'hover:-translate-y-0.5 hover:shadow-lg hover:border-ink-200/50',
};

// =============================================================================
// GLASS CARD COMPONENT
// =============================================================================

export const GlassCard = forwardRef<HTMLDivElement, GlassCardProps>(
    (
        {
            variant = 'light',
            padding = 'md',
            hover = false,
            border = true,
            children,
            className = '',
            ...props
        },
        ref
    ) => {
        return (
            <div
                ref={ref}
                className={`
          rounded-2xl
          transition-all duration-200 ease-out
          ${variantStyles[variant]}
          ${border ? borderStyles[variant] : ''}
          ${paddingStyles[padding]}
          ${hover ? `cursor-pointer ${hoverStyles[variant]}` : ''}
          ${className}
        `.trim().replace(/\s+/g, ' ')}
                {...props}
            >
                {children}
            </div>
        );
    }
);

GlassCard.displayName = 'GlassCard';

// =============================================================================
// GLASS PANEL
// =============================================================================

export interface GlassPanelProps extends GlassCardProps {
    /** Background opacity multiplier */
    opacity?: number;
}

export const GlassPanel: React.FC<GlassPanelProps> = ({
    children,
    className = '',
    ...props
}) => {
    return (
        <GlassCard
            className={`relative overflow-hidden ${className}`}
            {...props}
        >
            {/* Subtle shine effect */}
            <div className="absolute inset-0 bg-gradient-to-tr from-white/0 via-white/5 to-white/0 opacity-50 pointer-events-none" />
            <div className="relative z-10">
                {children}
            </div>
        </GlassCard>
    );
};

export default GlassCard;
