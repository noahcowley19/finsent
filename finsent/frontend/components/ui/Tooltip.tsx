'use client';

import React, { useState, useRef, useEffect, ReactNode, HTMLAttributes } from 'react';
import { createPortal } from 'react-dom';

// =============================================================================
// TYPES
// =============================================================================

export type TooltipPosition = 'top' | 'bottom' | 'left' | 'right';

/**
 * We Omit 'content' because HTMLAttributes defines it as a string (for <meta> tags).
 * This allows us to redefine it as a ReactNode for our UI component.
 */
export interface TooltipProps extends Omit<HTMLAttributes<HTMLDivElement>, 'content'> {
  /** Tooltip content - accepts text or JSX */
  content: ReactNode;
  /** Position relative to trigger */
  position?: TooltipPosition;
  /** Delay before showing (ms) */
  delay?: number;
  /** Disable tooltip */
  disabled?: boolean;
  /** Trigger element */
  children: ReactNode;
}

// =============================================================================
// COMPONENT
// =============================================================================

export const Tooltip: React.FC<TooltipProps> = ({
  content,
  position = 'top',
  delay = 200,
  disabled = false,
  children,
  className = '',
  ...props
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const [coords, setCoords] = useState({ top: 0, left: 0 });
  const triggerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<NodeJS.Timeout>();

  // Calculate position relative to the trigger element
  const updatePosition = () => {
    if (!triggerRef.current || !tooltipRef.current) return;

    const triggerRect = triggerRef.current.getBoundingClientRect();
    const tooltipRect = tooltipRef.current.getBoundingClientRect();
    const gap = 8;

    let top = 0;
    let left = 0;

    switch (position) {
      case 'top':
        top = triggerRect.top - tooltipRect.height - gap;
        left = triggerRect.left + (triggerRect.width - tooltipRect.width) / 2;
        break;
      case 'bottom':
        top = triggerRect.bottom + gap;
        left = triggerRect.left + (triggerRect.width - tooltipRect.width) / 2;
        break;
      case 'left':
        top = triggerRect.top + (triggerRect.height - tooltipRect.height) / 2;
        left = triggerRect.left - tooltipRect.width - gap;
        break;
      case 'right':
        top = triggerRect.top + (triggerRect.height - tooltipRect.height) / 2;
        left = triggerRect.right + gap;
        break;
    }

    // Clamp to viewport to prevent going off-screen
    const padding = 8;
    left = Math.max(padding, Math.min(left, window.innerWidth - tooltipRect.width - padding));
    top = Math.max(padding, Math.min(top, window.innerHeight - tooltipRect.height - padding));

    setCoords({ top, left });
  };

  const show = () => {
    if (disabled) return;
    timeoutRef.current = setTimeout(() => {
      setIsVisible(true);
    }, delay);
  };

  const hide = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  };

  // Lifecycle for position updates and event listeners
  useEffect(() => {
    if (isVisible) {
      updatePosition();
      window.addEventListener('scroll', updatePosition, true);
      window.addEventListener('resize', updatePosition);
    }

    return () => {
      window.removeEventListener('scroll', updatePosition, true);
      window.removeEventListener('resize', updatePosition);
    };
  }, [isVisible]);

  // Global cleanup
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);

  const arrowStyles: Record<TooltipPosition, string> = {
    top: 'bottom-0 left-1/2 -translate-x-1/2 translate-y-full border-t-navy-900 border-x-transparent border-b-transparent',
    bottom: 'top-0 left-1/2 -translate-x-1/2 -translate-y-full border-b-navy-900 border-x-transparent border-t-transparent',
    left: 'right-0 top-1/2 -translate-y-1/2 translate-x-full border-l-navy-900 border-y-transparent border-r-transparent',
    right: 'left-0 top-1/2 -translate-y-1/2 -translate-x-full border-r-navy-900 border-y-transparent border-l-transparent',
  };

  const tooltipElement = isVisible && typeof window !== 'undefined' ? (
    createPortal(
      <div
        ref={tooltipRef}
        role="tooltip"
        className={`
          fixed z-[9999]
          px-3 py-2
          bg-navy-900
          text-white text-sm
          rounded-md
          shadow-lg
          animate-in fade-in duration-200
          pointer-events-none
          max-w-xs
          ${className}
        `.trim().replace(/\s+/g, ' ')}
        style={{ top: coords.top, left: coords.left }}
        {...props}
      >
        {content}
        <div
          className={`absolute w-0 h-0 border-4 ${arrowStyles[position]}`}
          aria-hidden="true"
        />
      </div>,
      document.body
    )
  ) : null;

  return (
    <>
      <div
        ref={triggerRef}
        onMouseEnter={show}
        onMouseLeave={hide}
        onFocus={show}
        onBlur={hide}
        className="inline-block"
      >
        {children}
      </div>
      {tooltipElement}
    </>
  );
};

// =============================================================================
// INFO TOOLTIP (common pattern)
// =============================================================================

export interface InfoTooltipProps extends Omit<TooltipProps, 'children'> {
  /** Icon size */
  size?: 'sm' | 'md';
}

export const InfoTooltip: React.FC<InfoTooltipProps> = ({
  content,
  size = 'md',
  ...props
}) => {
  const sizeStyles = {
    sm: 'w-4 h-4',
    md: 'w-5 h-5',
  };

  return (
    <Tooltip content={content} {...props}>
      <button
        type="button"
        className={`
          ${sizeStyles[size]}
          text-slate-400
          hover:text-navy-500
          transition-colors duration-200
          focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy-500 rounded-full
        `}
        aria-label="More information"
      >
        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24" className="w-full h-full">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      </button>
    </Tooltip>
  );
};

export default Tooltip;
