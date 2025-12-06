'use client';

import { cn } from '@/lib/utils';

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg';
  className?: string;
}

const sizeClasses = {
  sm: 'w-4 h-4 border-2',
  md: 'w-8 h-8 border-3',
  lg: 'w-12 h-12 border-4',
};

export default function LoadingSpinner({ size = 'md', className }: LoadingSpinnerProps) {
  return (
    <div 
      className={cn(
        'rounded-full border-[var(--border)] border-t-[var(--accent)] animate-spin',
        sizeClasses[size],
        className
      )}
    />
  );
}

interface LoadingOverlayProps {
  message?: string;
}

export function LoadingOverlay({ message = 'Loading...' }: LoadingOverlayProps) {
  return (
    <div className="loading-overlay">
      {/* Animated logo */}
      <div 
        style={{
          marginBottom: '24px',
          animation: 'pulse 2s ease-in-out infinite',
        }}
      >
        <svg 
          width="56" 
          height="56" 
          viewBox="0 0 32 32" 
          fill="none"
        >
          <rect 
            x="2" 
            y="2" 
            width="28" 
            height="28" 
            rx="8" 
            fill="url(#loadingLogoGradient)"
          />
          <path 
            d="M10 22V14L16 10L22 14V22L16 26L10 22Z" 
            stroke="white" 
            strokeWidth="2" 
            strokeLinejoin="round"
            fill="none"
          />
          <circle cx="16" cy="16" r="3" fill="white" />
          <defs>
            <linearGradient id="loadingLogoGradient" x1="2" y1="2" x2="30" y2="30" gradientUnits="userSpaceOnUse">
              <stop stopColor="#00d4aa" />
              <stop offset="1" stopColor="#00a3ff" />
            </linearGradient>
          </defs>
        </svg>
      </div>
      
      {/* Spinner */}
      <div 
        style={{
          width: '40px',
          height: '40px',
          borderRadius: '50%',
          border: '3px solid var(--border)',
          borderTopColor: 'var(--accent)',
          animation: 'spin 1s linear infinite',
          marginBottom: '20px',
        }}
      />
      
      {/* Message */}
      <p 
        style={{
          color: 'var(--text-secondary)',
          fontSize: '15px',
          fontWeight: 500,
        }}
      >
        {message}
      </p>

      {/* Progress dots */}
      <div 
        style={{
          display: 'flex',
          gap: '8px',
          marginTop: '16px',
        }}
      >
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              background: 'var(--accent)',
              animation: `pulse 1.4s ease-in-out ${i * 0.2}s infinite`,
            }}
          />
        ))}
      </div>
    </div>
  );
}

interface LoadingCardProps {
  message?: string;
  className?: string;
}

export function LoadingCard({ message = 'Loading...', className }: LoadingCardProps) {
  return (
    <div className={cn('card flex flex-col items-center justify-center py-16', className)}>
      <div 
        style={{
          width: '36px',
          height: '36px',
          borderRadius: '50%',
          border: '3px solid var(--border)',
          borderTopColor: 'var(--accent)',
          animation: 'spin 1s linear infinite',
          marginBottom: '16px',
        }}
      />
      <p style={{ color: 'var(--text-secondary)', fontSize: '14px' }}>{message}</p>
    </div>
  );
}

// Skeleton loader for content placeholders
interface SkeletonProps {
  width?: string;
  height?: string;
  borderRadius?: string;
  className?: string;
}

export function Skeleton({ 
  width = '100%', 
  height = '20px', 
  borderRadius = '8px',
  className 
}: SkeletonProps) {
  return (
    <div
      className={className}
      style={{
        width,
        height,
        borderRadius,
        background: 'linear-gradient(90deg, var(--bg-secondary) 25%, var(--bg-tertiary) 50%, var(--bg-secondary) 75%)',
        backgroundSize: '200% 100%',
        animation: 'shimmer 1.5s ease-in-out infinite',
      }}
    />
  );
}
