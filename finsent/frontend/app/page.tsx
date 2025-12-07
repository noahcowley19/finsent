'use client';

import Link from 'next/link';
import { useEffect, useRef } from 'react';

// Feature icons as SVG components
const SearchIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="11" cy="11" r="8" />
    <path d="m21 21-4.3-4.3" />
  </svg>
);

const SentimentIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2v4" />
    <path d="m16.2 7.8 2.9-2.9" />
    <path d="M18 12h4" />
    <path d="m16.2 16.2 2.9 2.9" />
    <path d="M12 18v4" />
    <path d="m4.9 19.1 2.9-2.9" />
    <path d="M2 12h4" />
    <path d="m4.9 4.9 2.9 2.9" />
  </svg>
);

const FinancialsIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 3v18h18" />
    <path d="m19 9-5 5-4-4-3 3" />
  </svg>
);

const InsiderIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
    <circle cx="9" cy="7" r="4" />
    <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
    <path d="M16 3.13a4 4 0 0 1 0 7.75" />
  </svg>
);

const PortfolioIcon = () => (
  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <rect width="20" height="14" x="2" y="7" rx="2" ry="2" />
    <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
  </svg>
);

const ArrowIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M5 12h14" />
    <path d="m12 5 7 7-7 7" />
  </svg>
);

const features = [
  {
    title: 'Stock Search',
    description: 'Comprehensive stock analysis with interactive charts, real-time metrics, analyst ratings, and breaking news—all in one place.',
    href: '/search',
    icon: SearchIcon,
    featured: true,
  },
  {
    title: 'Sentiment Analyzer',
    description: 'Real-time market sentiment powered by transformer models, aggregating signals from X, StockTwits, and financial news.',
    href: '/sentiment',
    icon: SentimentIcon,
  },
  {
    title: 'Financial Analyzer',
    description: 'Academic scoring models including Piotroski F-Score, Altman Z-Score, and Beneish M-Score for deep fundamental analysis.',
    href: '/financials',
    icon: FinancialsIcon,
  },
  {
    title: 'Insider Trading',
    description: 'Track insider buying and selling patterns with cluster detection alerts and institutional ownership changes.',
    href: '/insider',
    icon: InsiderIcon,
  },
  {
    title: 'Portfolio Tracker',
    description: 'Monitor your investments with CAPM analysis, sector allocation, and projected returns based on your positions.',
    href: '/portfolio',
    icon: PortfolioIcon,
  },
];

// Animated particles component
function Particles() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let particles: Array<{
      x: number;
      y: number;
      size: number;
      speedX: number;
      speedY: number;
      opacity: number;
    }> = [];

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const createParticles = () => {
      particles = [];
      const particleCount = Math.floor((canvas.width * canvas.height) / 15000);
      
      for (let i = 0; i < particleCount; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          size: Math.random() * 2 + 0.5,
          speedX: (Math.random() - 0.5) * 0.3,
          speedY: (Math.random() - 0.5) * 0.3,
          opacity: Math.random() * 0.5 + 0.1,
        });
      }
    };

    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((particle, index) => {
        particle.x += particle.speedX;
        particle.y += particle.speedY;

        // Wrap around edges
        if (particle.x < 0) particle.x = canvas.width;
        if (particle.x > canvas.width) particle.x = 0;
        if (particle.y < 0) particle.y = canvas.height;
        if (particle.y > canvas.height) particle.y = 0;

        // Draw particle
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 212, 170, ${particle.opacity})`;
        ctx.fill();

        // Draw connections
        particles.forEach((otherParticle, otherIndex) => {
          if (index === otherIndex) return;
          
          const dx = particle.x - otherParticle.x;
          const dy = particle.y - otherParticle.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 120) {
            ctx.beginPath();
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(otherParticle.x, otherParticle.y);
            ctx.strokeStyle = `rgba(0, 212, 170, ${0.1 * (1 - distance / 120)})`;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
        });
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    resize();
    createParticles();
    animate();

    window.addEventListener('resize', () => {
      resize();
      createParticles();
    });

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 0,
        opacity: 0.6,
      }}
    />
  );
}

// Feature card with mouse tracking
function FeatureCard({ 
  feature, 
  index 
}: { 
  feature: typeof features[0]; 
  index: number;
}) {
  const cardRef = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    cardRef.current.style.setProperty('--mouse-x', `${x}%`);
    cardRef.current.style.setProperty('--mouse-y', `${y}%`);
  };

  const Icon = feature.icon;

  return (
    <div
      ref={cardRef}
      className={`feature-card ${feature.featured ? 'featured' : ''}`}
      onMouseMove={handleMouseMove}
      style={{
        animation: `fadeInUp 0.8s var(--ease-out-expo) ${0.3 + index * 0.1}s both`,
      }}
    >
      <div className="feature-icon">
        <Icon />
      </div>
      <h2>{feature.title}</h2>
      <p>{feature.description}</p>
      <Link href={feature.href} className="feature-btn">
        Launch Tool
        <ArrowIcon />
      </Link>
    </div>
  );
}

export default function Home() {
  return (
    <>
      {/* Animated background */}
      <div className="mesh-gradient-bg" />
      <div className="grid-overlay" />
      <Particles />

      <div className="container" style={{ position: 'relative', zIndex: 1 }}>
        {/* Hero Section */}
        <div className="hero">
          <div className="hero-badge">
            <span>Financial Intelligence Platform</span>
          </div>
          
          <h1>
            Invest Smarter with{' '}
            <span className="gradient-text">Caveray</span>
          </h1>
          
          <p className="subtitle">
            Professional-grade financial analysis tools combining real-time sentiment tracking, 
            academic scoring models, and insider activity monitoring.
          </p>
        </div>

        {/* Features Grid */}
        <div className="features-grid">
          {features.map((feature, index) => (
            <FeatureCard key={feature.href} feature={feature} index={index} />
          ))}
        </div>

        {/* Stats Section */}
        <div 
          style={{ 
            display: 'flex', 
            justifyContent: 'center', 
            gap: '60px', 
            marginTop: '80px',
            paddingTop: '60px',
            borderTop: '1px solid var(--border)',
            animation: 'fadeInUp 0.8s var(--ease-out-expo) 0.8s both',
          }}
        >
          {[
            { value: '5', label: 'Analysis Tools' },
            { value: '3', label: 'Scoring Models' },
            { value: '∞', label: 'Insights' },
          ].map((stat, index) => (
            <div key={index} style={{ textAlign: 'center' }}>
              <div 
                style={{ 
                  fontSize: '2.5rem', 
                  fontWeight: 800, 
                  color: 'var(--accent)',
                  fontFamily: "'JetBrains Mono', monospace",
                  letterSpacing: '-0.03em',
                }}
              >
                {stat.value}
              </div>
              <div 
                style={{ 
                  fontSize: '14px', 
                  color: 'var(--text-tertiary)',
                  fontWeight: 500,
                  marginTop: '4px',
                }}
              >
                {stat.label}
              </div>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}
