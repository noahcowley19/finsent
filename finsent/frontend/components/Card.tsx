import { cn } from '@/lib/utils';
import { ReactNode } from 'react';

interface CardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
  featured?: boolean;
  compact?: boolean;
}

export default function Card({ 
  children, 
  className,
  hover = false,
  featured = false,
  compact = false
}: CardProps) {
  return (
    <div 
      className={cn(
        'card',
        hover && 'card-hover',
        featured && 'card-featured',
        compact && 'card-compact',
        className
      )}
    >
      {children}
    </div>
  );
}

interface CardHeaderProps {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  className?: string;
}

export function CardHeader({ title, subtitle, action, className }: CardHeaderProps) {
  return (
    <div className={cn('mb-6', className)}>
      <div className="flex items-start justify-between gap-4">
        <div>
          <h3 
            className="text-xl font-bold text-primary"
            style={{ letterSpacing: '-0.02em' }}
          >
            {title}
          </h3>
          {subtitle && (
            <p className="text-sm text-secondary mt-1">{subtitle}</p>
          )}
        </div>
        {action && <div>{action}</div>}
      </div>
    </div>
  );
}

interface CardSectionProps {
  title?: string;
  children: ReactNode;
  className?: string;
}

export function CardSection({ title, children, className }: CardSectionProps) {
  return (
    <div className={cn('mb-6 last:mb-0', className)}>
      {title && (
        <h4 className="section-title">{title}</h4>
      )}
      {children}
    </div>
  );
}
