'use client';

// =============================================================================
// ANIMATED COUNTER COMPONENT
// =============================================================================
// Smooth count-up animation for numbers on page load
//
// Location: frontend/components/ui/AnimatedCounter.tsx
// =============================================================================

import React, { useState, useEffect, useRef } from 'react';

interface AnimatedCounterProps {
    /** Target value to count to */
    value: number;
    /** Duration of animation in ms */
    duration?: number;
    /** Format as currency */
    currency?: boolean;
    /** Format as percentage */
    percentage?: boolean;
    /** Decimal places */
    decimals?: number;
    /** Prefix string */
    prefix?: string;
    /** Suffix string */
    suffix?: string;
    /** Additional className */
    className?: string;
}

const easeOutExpo = (t: number): number => {
    return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
};

export const AnimatedCounter: React.FC<AnimatedCounterProps> = ({
    value,
    duration = 2000,
    currency = false,
    percentage = false,
    decimals = 0,
    prefix = '',
    suffix = '',
    className = '',
}) => {
    const [displayValue, setDisplayValue] = useState(0);
    const startRef = useRef<number | null>(null);
    const rafRef = useRef<number | null>(null);
    const observerRef = useRef<IntersectionObserver | null>(null);
    const elementRef = useRef<HTMLSpanElement | null>(null);
    const [hasAnimated, setHasAnimated] = useState(false);

    const formatValue = (num: number): string => {
        let formatted: string;

        if (currency) {
            formatted = num.toLocaleString('en-US', {
                minimumFractionDigits: decimals,
                maximumFractionDigits: decimals,
            });
            return `${prefix}$${formatted}${suffix}`;
        }

        if (percentage) {
            formatted = num.toLocaleString('en-US', {
                minimumFractionDigits: decimals,
                maximumFractionDigits: decimals,
            });
            return `${prefix}${formatted}%${suffix}`;
        }

        // Handle large numbers with abbreviations
        if (num >= 1000000) {
            formatted = (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            formatted = (num / 1000).toFixed(1) + 'K';
        } else {
            formatted = num.toLocaleString('en-US', {
                minimumFractionDigits: decimals,
                maximumFractionDigits: decimals,
            });
        }

        return `${prefix}${formatted}${suffix}`;
    };

    const animate = (timestamp: number) => {
        if (startRef.current === null) {
            startRef.current = timestamp;
        }

        const elapsed = timestamp - startRef.current;
        const progress = Math.min(elapsed / duration, 1);
        const easedProgress = easeOutExpo(progress);

        setDisplayValue(Math.floor(easedProgress * value));

        if (progress < 1) {
            rafRef.current = requestAnimationFrame(animate);
        } else {
            setDisplayValue(value);
        }
    };

    useEffect(() => {
        // Use Intersection Observer to trigger animation when visible
        observerRef.current = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting && !hasAnimated) {
                        setHasAnimated(true);
                        rafRef.current = requestAnimationFrame(animate);
                    }
                });
            },
            { threshold: 0.1 }
        );

        if (elementRef.current) {
            observerRef.current.observe(elementRef.current);
        }

        return () => {
            if (rafRef.current) {
                cancelAnimationFrame(rafRef.current);
            }
            if (observerRef.current) {
                observerRef.current.disconnect();
            }
        };
    }, [value, duration, hasAnimated]);

    return (
        <span ref={elementRef} className={className}>
            {formatValue(displayValue)}
        </span>
    );
};

export default AnimatedCounter;
