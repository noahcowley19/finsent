'use client';

import React from 'react';

// =============================================================================
// TYPES
// =============================================================================

export interface GlassCardProps {
    /** Card content */
    children: React.ReactNode;
    /** Additional CSS classes */
    className?: string;
    /** Use glass effect (default) or solid white */
    variant?: 'glass' | 'solid';
    /** Add hover effect */
    hoverable?: boolean;
    /** Click handler */
    onClick?: () => void;
}

// =============================================================================
// GLASS CARD COMPONENT - Editorial Finance Style
// =============================================================================

export const GlassCard: React.FC<GlassCardProps> = ({
    children,
    className = '',
    variant = 'solid',
    hoverable = false,
    onClick,
}) => {
    const baseStyles = 'rounded-xl p-6 border border-gray-200';

    const variantStyles = {
        glass: 'bg-white/60 backdrop-blur-sm',
        solid: 'bg-white',
    };

    const hoverStyles = hoverable
        ? 'transition-all duration-200 hover:border-gray-300 hover:shadow-sm cursor-pointer'
        : '';

    return (
        <div
            className={`${baseStyles} ${variantStyles[variant]} ${hoverStyles} ${className}`}
            onClick={onClick}
        >
            {children}
        </div>
    );
};

export default GlassCard;
