'use client';

import React, { useRef, useState, ReactNode, MouseEvent, useCallback } from 'react';
import Link from 'next/link';

// =============================================================================
// MAGNETIC BUTTON COMPONENT
// =============================================================================
// Button with "magnetic pull" effect where the button follows cursor within
// a 10px radius, combined with scale(1.02) on hover.
// =============================================================================

export interface MagneticButtonProps {
    children: ReactNode;
    href?: string;
    onClick?: () => void;
    variant?: 'primary' | 'secondary' | 'ghost';
    size?: 'sm' | 'md' | 'lg';
    className?: string;
    disabled?: boolean;
    type?: 'button' | 'submit' | 'reset';
    icon?: ReactNode;
    iconPosition?: 'left' | 'right';
}

const sizeClasses = {
    sm: 'h-9 px-4 text-sm rounded-lg',
    md: 'h-11 px-6 text-sm rounded-xl',
    lg: 'h-13 px-8 text-base rounded-xl',
};

const variantClasses = {
    primary: 'bg-obsidian-900 text-white border-transparent hover:bg-obsidian-850',
    secondary: 'bg-transparent text-obsidian-900 border border-obsidian-200 hover:bg-cream-100 hover:border-obsidian-300',
    ghost: 'bg-transparent text-obsidian-600 hover:bg-cream-100 hover:text-obsidian-900',
};

export const MagneticButton: React.FC<MagneticButtonProps> = ({
    children,
    href,
    onClick,
    variant = 'primary',
    size = 'md',
    className = '',
    disabled = false,
    type = 'button',
    icon,
    iconPosition = 'right',
}) => {
    const buttonRef = useRef<HTMLButtonElement | HTMLAnchorElement>(null);
    const [transform, setTransform] = useState({ x: 0, y: 0, scale: 1 });

    const handleMouseMove = useCallback((e: MouseEvent) => {
        if (disabled || !buttonRef.current) return;

        const rect = buttonRef.current.getBoundingClientRect();
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;

        const deltaX = e.clientX - centerX;
        const deltaY = e.clientY - centerY;

        // Limit magnetic pull to 10px radius
        const maxOffset = 10;
        const distance = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
        const clampedDistance = Math.min(distance, maxOffset);
        const factor = clampedDistance / Math.max(distance, 1);

        setTransform({
            x: deltaX * factor * 0.3,
            y: deltaY * factor * 0.3,
            scale: 1.02,
        });
    }, [disabled]);

    const handleMouseLeave = useCallback(() => {
        setTransform({ x: 0, y: 0, scale: 1 });
    }, []);

    const baseClasses = `
    inline-flex items-center justify-center gap-2 font-medium
    transition-colors duration-150 ease-out
    ${sizeClasses[size]}
    ${variantClasses[variant]}
    ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
    ${className}
  `;

    const style = {
        transform: `translate(${transform.x}px, ${transform.y}px) scale(${transform.scale})`,
        transition: 'transform 0.2s cubic-bezier(0.16, 1, 0.3, 1)',
    };

    const content = (
        <>
            {icon && iconPosition === 'left' && icon}
            <span>{children}</span>
            {icon && iconPosition === 'right' && icon}
        </>
    );

    if (href && !disabled) {
        return (
            <Link
                href={href}
                ref={buttonRef as React.Ref<HTMLAnchorElement>}
                className={baseClasses}
                style={style}
                onMouseMove={handleMouseMove as any}
                onMouseLeave={handleMouseLeave}
            >
                {content}
            </Link>
        );
    }

    return (
        <button
            ref={buttonRef as React.Ref<HTMLButtonElement>}
            type={type}
            onClick={onClick}
            disabled={disabled}
            className={baseClasses}
            style={style}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
        >
            {content}
        </button>
    );
};

// =============================================================================
// ARROW ICON
// =============================================================================

export const ArrowIcon: React.FC<{ className?: string }> = ({ className = 'w-4 h-4' }) => (
    <svg className={className} fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
    </svg>
);

export default MagneticButton;
