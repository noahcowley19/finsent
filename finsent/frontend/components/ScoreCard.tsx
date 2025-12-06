import { cn } from '@/lib/utils';

type ScoreStatus = 'positive' | 'negative' | 'neutral' | 'warning';

interface ScoreDetail {
  label: string;
  value: string | number;
  passed?: boolean;
}

interface ScoreCardProps {
  title: string;
  score: string | number;
  maxScore?: number;
  interpretation: string;
  status: ScoreStatus;
  details?: ScoreDetail[];
  className?: string;
}

export default function ScoreCard({
  title,
  score,
  maxScore,
  interpretation,
  status,
  details,
  className
}: ScoreCardProps) {
  return (
    <div className={cn('card', className)}>
      <h3 
        className="text-lg font-bold text-primary mb-4"
        style={{ letterSpacing: '-0.02em' }}
      >
        {title}
      </h3>
      
      <div className={cn('score-display', status)}>
        <div className="score-value">
          {score}
          {maxScore && (
            <span className="text-2xl opacity-60">/{maxScore}</span>
          )}
        </div>
        <div className="score-label">{interpretation}</div>
      </div>

      {details && details.length > 0 && (
        <div className="space-y-2 mt-4">
          {details.map((detail, i) => (
            <div 
              key={i} 
              className="flex justify-between items-center py-2 px-3 bg-background rounded-lg text-sm"
            >
              <span className="text-secondary">{detail.label}</span>
              <span className={cn(
                'font-medium',
                detail.passed === true && 'text-positive',
                detail.passed === false && 'text-negative',
                detail.passed === undefined && 'text-primary'
              )}>
                {detail.value}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

interface ScoreBarProps {
  value: number;
  max: number;
  status?: ScoreStatus;
  showLabel?: boolean;
  className?: string;
}

export function ScoreBar({ 
  value, 
  max, 
  status = 'neutral',
  showLabel = true,
  className 
}: ScoreBarProps) {
  const percentage = Math.min((value / max) * 100, 100);
  
  const statusColors: Record<ScoreStatus, string> = {
    positive: 'bg-positive',
    negative: 'bg-negative',
    neutral: 'bg-neutral',
    warning: 'bg-warning',
  };

  return (
    <div className={cn('w-full', className)}>
      {showLabel && (
        <div className="flex justify-between text-sm mb-2">
          <span className="text-secondary">Score</span>
          <span className="font-semibold text-primary">{value}/{max}</span>
        </div>
      )}
      <div className="h-3 bg-neutral-light rounded-full overflow-hidden">
        <div 
          className={cn('h-full rounded-full transition-all duration-500', statusColors[status])}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}
