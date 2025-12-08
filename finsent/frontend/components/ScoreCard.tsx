'use client';

interface ScoreCardProps {
  title: string;
  score: number | null;
  maxScore?: number;
  display: string;
  interpretation: string;
  status: 'positive' | 'negative' | 'neutral';
  description?: string;
  className?: string;
}

export default function ScoreCard({ 
  title, 
  score, 
  maxScore = 10, 
  display, 
  interpretation, 
  status,
  description,
  className = '' 
}: ScoreCardProps) {
  const statusColors = {
    positive: { color: 'var(--positive)', bg: 'var(--positive-light)', gradient: 'linear-gradient(135deg, var(--positive), #00b38a)' },
    negative: { color: 'var(--negative)', bg: 'var(--negative-light)', gradient: 'linear-gradient(135deg, var(--negative), #ff4f4f)' },
    neutral: { color: 'var(--warning)', bg: 'var(--warning-light)', gradient: 'linear-gradient(135deg, var(--warning), #f59e0b)' },
  };

  const colors = statusColors[status];
  const percentage = score !== null ? (score / maxScore) * 100 : 0;

  return (
    <div
      className={className}
      style={{
        background: 'var(--bg-card)',
        borderRadius: '20px',
        padding: '24px',
        border: '1px solid var(--border)',
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      {/* Glow effect */}
      <div
        style={{
          position: 'absolute',
          top: '-50%',
          right: '-50%',
          width: '100%',
          height: '100%',
          background: `radial-gradient(circle, ${colors.bg} 0%, transparent 70%)`,
          opacity: 0.5,
          pointerEvents: 'none',
        }}
      />

      <div style={{ position: 'relative', zIndex: 1 }}>
        {/* Title */}
        <div
          style={{
            fontSize: '12px',
            fontWeight: 600,
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            color: 'var(--text-tertiary)',
            marginBottom: '16px',
          }}
        >
          {title}
        </div>

        {/* Score Display */}
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '8px', marginBottom: '16px' }}>
          <span
            style={{
              fontSize: '2.5rem',
              fontWeight: 800,
              fontFamily: "'JetBrains Mono', monospace",
              background: colors.gradient,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              lineHeight: 1,
            }}
          >
            {display}
          </span>
          {maxScore && (
            <span style={{ fontSize: '14px', color: 'var(--text-muted)', marginBottom: '6px' }}>
              / {maxScore}
            </span>
          )}
        </div>

        {/* Progress Bar */}
        {score !== null && (
          <div style={{ marginBottom: '16px' }}>
            <div
              style={{
                height: '6px',
                background: 'var(--bg-secondary)',
                borderRadius: '3px',
                overflow: 'hidden',
              }}
            >
              <div
                style={{
                  height: '100%',
                  width: `${percentage}%`,
                  background: colors.gradient,
                  borderRadius: '3px',
                  transition: 'width 0.5s ease-out',
                }}
              />
            </div>
          </div>
        )}

        {/* Interpretation Badge */}
        <div
          style={{
            display: 'inline-block',
            padding: '6px 14px',
            borderRadius: '8px',
            background: colors.bg,
            color: colors.color,
            fontSize: '13px',
            fontWeight: 600,
            marginBottom: description ? '12px' : 0,
          }}
        >
          {interpretation}
        </div>

        {/* Description */}
        {description && (
          <p style={{ fontSize: '13px', color: 'var(--text-tertiary)', margin: 0, lineHeight: 1.6 }}>
            {description}
          </p>
        )}
      </div>
    </div>
  );
}

interface ScoreBarProps {
  label: string;
  value: number;
  maxValue?: number;
  status?: 'positive' | 'negative' | 'neutral';
}

export function ScoreBar({ label, value, maxValue = 100, status = 'neutral' }: ScoreBarProps) {
  const statusColors = {
    positive: 'var(--positive)',
    negative: 'var(--negative)',
    neutral: 'var(--accent)',
  };

  const percentage = Math.min((value / maxValue) * 100, 100);

  return (
    <div style={{ marginBottom: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
        <span style={{ fontSize: '13px', color: 'var(--text-secondary)' }}>{label}</span>
        <span style={{ fontSize: '13px', fontWeight: 600, fontFamily: "'JetBrains Mono', monospace" }}>
          {value}
        </span>
      </div>
      <div
        style={{
          height: '8px',
          background: 'var(--bg-secondary)',
          borderRadius: '4px',
          overflow: 'hidden',
        }}
      >
        <div
          style={{
            height: '100%',
            width: `${percentage}%`,
            background: statusColors[status],
            borderRadius: '4px',
            transition: 'width 0.4s ease-out',
          }}
        />
      </div>
    </div>
  );
}
