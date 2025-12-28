'use client';

import React, { useEffect, useRef, useState, ReactNode } from 'react';

// =============================================================================
// SCROLL REVEAL COMPONENT
// =============================================================================
// Wrapper component that triggers fade-in-up animation when element enters
// viewport with 100px offset threshold.
// =============================================================================

export interface ScrollRevealProps {
    children: ReactNode;
    className?: string;
    delay?: number;
    threshold?: number;
    triggerOnce?: boolean;
}

export const ScrollReveal: React.FC<ScrollRevealProps> = ({
    children,
    className = '',
    delay = 0,
    threshold = 0.1,
    triggerOnce = true,
}) => {
    const ref = useRef<HTMLDivElement>(null);
    const [isRevealed, setIsRevealed] = useState(false);

    useEffect(() => {
        const element = ref.current;
        if (!element) return;

        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setIsRevealed(true);
                    if (triggerOnce) {
                        observer.disconnect();
                    }
                } else if (!triggerOnce) {
                    setIsRevealed(false);
                }
            },
            {
                threshold,
                rootMargin: '-100px 0px',
            }
        );

        observer.observe(element);

        return () => observer.disconnect();
    }, [threshold, triggerOnce]);

    return (
        <div
            ref={ref}
            className={`scroll-reveal ${isRevealed ? 'revealed' : ''} ${className}`}
            style={{ transitionDelay: `${delay}ms` }}
        >
            {children}
        </div>
    );
};

// =============================================================================
// USE SCROLL REVEAL HOOK
// =============================================================================
// Hook version for more flexible usage in custom components.
// =============================================================================

export interface UseScrollRevealOptions {
    threshold?: number;
    rootMargin?: string;
    triggerOnce?: boolean;
}

export const useScrollReveal = (options: UseScrollRevealOptions = {}) => {
    const { threshold = 0.1, rootMargin = '-100px 0px', triggerOnce = true } = options;
    const ref = useRef<HTMLElement>(null);
    const [isRevealed, setIsRevealed] = useState(false);

    useEffect(() => {
        const element = ref.current;
        if (!element) return;

        const observer = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    setIsRevealed(true);
                    if (triggerOnce) {
                        observer.disconnect();
                    }
                } else if (!triggerOnce) {
                    setIsRevealed(false);
                }
            },
            { threshold, rootMargin }
        );

        observer.observe(element);

        return () => observer.disconnect();
    }, [threshold, rootMargin, triggerOnce]);

    return { ref, isRevealed };
};

export default ScrollReveal;
