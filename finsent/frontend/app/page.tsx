'use client';

import Link from 'next/link';
import { useEffect, useRef, useState } from 'react';

// ============================================
// ANIMATED GRADIENT ORB COMPONENT
// ============================================
function GradientOrbs() {
  return (
    <div className="gradient-orbs">
      <div className="orb orb-1" />
      <div className="orb orb-2" />
      <div className="orb orb-3" />
      <div className="orb orb-4" />
    </div>
  );
}

// ============================================
// ADVANCED PARTICLE NETWORK SYSTEM
// ============================================
function ParticleNetwork() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mouseRef = useRef({ x: 0, y: 0 });
  const particlesRef = useRef<Array<{
    x: number;
    y: number;
    vx: number;
    vy: number;
    size: number;
    opacity: number;
    pulsePhase: number;
    connections: number[];
  }>>([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationFrameId: number;
    let particles = particlesRef.current;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      initParticles();
    };

    const initParticles = () => {
      particles = [];
      const particleCount = Math.min(Math.floor((canvas.width * canvas.height) / 12000), 150);
      
      for (let i = 0; i < particleCount; i++) {
        particles.push({
          x: Math.random() * canvas.width,
          y: Math.random() * canvas.height,
          vx: (Math.random() - 0.5) * 0.4,
          vy: (Math.random() - 0.5) * 0.4,
          size: Math.random() * 2 + 1,
          opacity: Math.random() * 0.5 + 0.2,
          pulsePhase: Math.random() * Math.PI * 2,
          connections: [],
        });
      }
      particlesRef.current = particles;
    };

    const handleMouseMove = (e: MouseEvent) => {
      mouseRef.current = { x: e.clientX, y: e.clientY };
    };

    const animate = (time: number) => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      particles.forEach((particle, i) => {
        // Mouse attraction
        const dx = mouseRef.current.x - particle.x;
        const dy = mouseRef.current.y - particle.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        
        if (dist < 200) {
          const force = (200 - dist) / 200 * 0.02;
          particle.vx += dx * force * 0.01;
          particle.vy += dy * force * 0.01;
        }

        // Apply velocity with damping
        particle.x += particle.vx;
        particle.y += particle.vy;
        particle.vx *= 0.99;
        particle.vy *= 0.99;

        // Wrap around edges
        if (particle.x < 0) particle.x = canvas.width;
        if (particle.x > canvas.width) particle.x = 0;
        if (particle.y < 0) particle.y = canvas.height;
        if (particle.y > canvas.height) particle.y = 0;

        // Pulse effect
        const pulse = Math.sin(time * 0.002 + particle.pulsePhase) * 0.3 + 0.7;
        const currentOpacity = particle.opacity * pulse;

        // Draw glow
        const gradient = ctx.createRadialGradient(
          particle.x, particle.y, 0,
          particle.x, particle.y, particle.size * 4
        );
        gradient.addColorStop(0, `rgba(0, 212, 170, ${currentOpacity * 0.8})`);
        gradient.addColorStop(0.5, `rgba(0, 212, 170, ${currentOpacity * 0.2})`);
        gradient.addColorStop(1, 'rgba(0, 212, 170, 0)');
        
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size * 4, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Draw core
        ctx.beginPath();
        ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 212, 170, ${currentOpacity})`;
        ctx.fill();

        // Draw connections
        particles.forEach((other, j) => {
          if (i >= j) return;
          
          const cdx = particle.x - other.x;
          const cdy = particle.y - other.y;
          const distance = Math.sqrt(cdx * cdx + cdy * cdy);

          if (distance < 150) {
            const opacity = (1 - distance / 150) * 0.15 * pulse;
            
            ctx.beginPath();
            ctx.moveTo(particle.x, particle.y);
            ctx.lineTo(other.x, other.y);
            
            const lineGradient = ctx.createLinearGradient(
              particle.x, particle.y, other.x, other.y
            );
            lineGradient.addColorStop(0, `rgba(0, 212, 170, ${opacity})`);
            lineGradient.addColorStop(0.5, `rgba(0, 163, 255, ${opacity * 0.8})`);
            lineGradient.addColorStop(1, `rgba(0, 212, 170, ${opacity})`);
            
            ctx.strokeStyle = lineGradient;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        });
      });

      animationFrameId = requestAnimationFrame(animate);
    };

    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('mousemove', handleMouseMove);
    animationFrameId = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(animationFrameId);
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMouseMove);
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
        zIndex: 1,
      }}
    />
  );
}

// ============================================
// MAGNETIC BUTTON COMPONENT
// ============================================
function MagneticButton({ 
  children, 
  href, 
  className = '',
  variant = 'primary'
}: { 
  children: React.ReactNode;
  href: string;
  className?: string;
  variant?: 'primary' | 'secondary';
}) {
  const buttonRef = useRef<HTMLAnchorElement>(null);
  const [position, setPosition] = useState({ x: 0, y: 0 });

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!buttonRef.current) return;
    const rect = buttonRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left - rect.width / 2;
    const y = e.clientY - rect.top - rect.height / 2;
    setPosition({ x: x * 0.3, y: y * 0.3 });
  };

  const handleMouseLeave = () => {
    setPosition({ x: 0, y: 0 });
  };

  return (
    <Link
      ref={buttonRef}
      href={href}
      className={`magnetic-btn magnetic-btn-${variant} ${className}`}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{
        transform: `translate(${position.x}px, ${position.y}px)`,
      }}
    >
      <span className="magnetic-btn-bg" />
      <span className="magnetic-btn-content">{children}</span>
    </Link>
  );
}

// ============================================
// ANIMATED COUNTER COMPONENT
// ============================================
function AnimatedCounter({ 
  value, 
  suffix = '',
  duration = 2000 
}: { 
  value: string | number;
  suffix?: string;
  duration?: number;
}) {
  const [count, setCount] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting && !isVisible) {
          setIsVisible(true);
        }
      },
      { threshold: 0.5 }
    );

    if (ref.current) observer.observe(ref.current);
    return () => observer.disconnect();
  }, [isVisible]);

  useEffect(() => {
    if (!isVisible) return;
    
    const numValue = typeof value === 'string' ? parseFloat(value) || 0 : value;
    if (isNaN(numValue)) return;

    let startTime: number;
    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime;
      const progress = Math.min((currentTime - startTime) / duration, 1);
      
      // Easing function
      const easeOutExpo = 1 - Math.pow(2, -10 * progress);
      setCount(Math.floor(numValue * easeOutExpo));

      if (progress < 1) {
        requestAnimationFrame(animate);
      } else {
        setCount(numValue);
      }
    };

    requestAnimationFrame(animate);
  }, [isVisible, value, duration]);

  const displayValue = typeof value === 'string' && isNaN(parseFloat(value)) 
    ? value 
    : count;

  return (
    <span ref={ref} className="animated-counter">
      {displayValue}{suffix}
    </span>
  );
}

// ============================================
// TILT CARD COMPONENT
// ============================================
function TiltCard({ 
  children, 
  className = '',
  glowColor = 'rgba(0, 212, 170, 0.15)'
}: { 
  children: React.ReactNode;
  className?: string;
  glowColor?: string;
}) {
  const cardRef = useRef<HTMLDivElement>(null);
  const [transform, setTransform] = useState('');
  const [glowPosition, setGlowPosition] = useState({ x: 50, y: 50 });

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!cardRef.current) return;
    
    const rect = cardRef.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    
    const tiltX = (y - 0.5) * 10;
    const tiltY = (x - 0.5) * -10;
    
    setTransform(`perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) scale3d(1.02, 1.02, 1.02)`);
    setGlowPosition({ x: x * 100, y: y * 100 });
  };

  const handleMouseLeave = () => {
    setTransform('perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)');
  };

  return (
    <div
      ref={cardRef}
      className={`tilt-card ${className}`}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      style={{ transform }}
    >
      <div 
        className="tilt-card-glow"
        style={{
          background: `radial-gradient(circle at ${glowPosition.x}% ${glowPosition.y}%, ${glowColor} 0%, transparent 50%)`,
        }}
      />
      <div className="tilt-card-content">
        {children}
      </div>
    </div>
  );
}

// ============================================
// ANIMATED TEXT REVEAL
// ============================================
function TextReveal({ 
  children, 
  delay = 0 
}: { 
  children: string;
  delay?: number;
}) {
  const words = children.split(' ');
  
  return (
    <span className="text-reveal">
      {words.map((word, i) => (
        <span 
          key={i} 
          className="text-reveal-word"
          style={{ animationDelay: `${delay + i * 0.05}s` }}
        >
          {word}&nbsp;
        </span>
      ))}
    </span>
  );
}

// ============================================
// FEATURE ICONS
// ============================================
const icons = {
  search: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" />
      <path d="m21 21-4.3-4.3" />
      <path d="M11 8v6M8 11h6" />
    </svg>
  ),
  sentiment: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a10 10 0 1 0 10 10" />
      <path d="M12 12 2 12" />
      <path d="M12 2v10" />
      <path d="m17 7 5-5" />
      <path d="M22 2h-5v5" />
    </svg>
  ),
  financials: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 3v18h18" />
      <path d="m7 16 4-4 4 4 6-6" />
      <circle cx="21" cy="10" r="1" fill="currentColor" />
    </svg>
  ),
  insider: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  ),
  portfolio: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="7" width="20" height="14" rx="2" />
      <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
      <path d="M6 12h.01M12 12h.01M18 12h.01" />
    </svg>
  ),
  arrow: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M5 12h14" />
      <path d="m12 5 7 7-7 7" />
    </svg>
  ),
};

// ============================================
// FEATURE DATA
// ============================================
const features = [
  {
    id: 'search',
    title: 'Stock Search',
    description: 'Comprehensive analysis with real-time charts, metrics, analyst ratings, and breaking news.',
    href: '/search',
    icon: icons.search,
    gradient: 'linear-gradient(135deg, #00d4aa 0%, #00a3ff 100%)',
    featured: true,
    stats: '10K+ Stocks',
  },
  {
    id: 'sentiment',
    title: 'Sentiment Analyzer',
    description: 'Real-time market sentiment from X, StockTwits, and news powered by transformer models.',
    href: '/sentiment',
    icon: icons.sentiment,
    gradient: 'linear-gradient(135deg, #8b5cf6 0%, #d946ef 100%)',
    stats: 'Live Data',
  },
  {
    id: 'financials',
    title: 'Financial Analyzer',
    description: 'Academic scoring models: Piotroski F-Score, Altman Z-Score, and Beneish M-Score.',
    href: '/financials',
    icon: icons.financials,
    gradient: 'linear-gradient(135deg, #f59e0b 0%, #ef4444 100%)',
    stats: '3 Models',
  },
  {
    id: 'insider',
    title: 'Insider Trading',
    description: 'Track insider buying/selling patterns with cluster detection and ownership changes.',
    href: '/insider',
    icon: icons.insider,
    gradient: 'linear-gradient(135deg, #06b6d4 0%, #3b82f6 100%)',
    stats: 'SEC Data',
  },
  {
    id: 'portfolio',
    title: 'Portfolio Tracker',
    description: 'Monitor investments with CAPM analysis, sector allocation, and projected returns.',
    href: '/portfolio',
    icon: icons.portfolio,
    gradient: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
    stats: 'Real-time',
  },
];

// ============================================
// FEATURE CARD COMPONENT
// ============================================
function FeatureCard({ 
  feature, 
  index 
}: { 
  feature: typeof features[0];
  index: number;
}) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <TiltCard 
      className={`feature-card-wrapper ${feature.featured ? 'featured' : ''}`}
      glowColor={feature.featured ? 'rgba(0, 212, 170, 0.2)' : 'rgba(0, 212, 170, 0.1)'}
    >
      <Link 
        href={feature.href}
        className={`feature-card ${feature.featured ? 'featured' : ''}`}
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
        style={{ animationDelay: `${0.2 + index * 0.1}s` }}
      >
        {/* Animated border gradient */}
        <div className="feature-card-border" />
        
        {/* Icon with gradient background */}
        <div 
          className="feature-icon-wrapper"
          style={{ 
            background: isHovered ? feature.gradient : 'var(--bg-elevated)',
          }}
        >
          <div className="feature-icon">
            {feature.icon}
          </div>
        </div>

        {/* Content */}
        <div className="feature-content">
          <div className="feature-header">
            <h3>{feature.title}</h3>
            <span className="feature-stat">{feature.stats}</span>
          </div>
          <p>{feature.description}</p>
        </div>

        {/* Arrow indicator */}
        <div className="feature-arrow">
          {icons.arrow}
        </div>

        {/* Shine effect on hover */}
        <div className="feature-shine" />
      </Link>
    </TiltCard>
  );
}

// ============================================
// FLOATING BADGE COMPONENT
// ============================================
function FloatingBadge() {
  return (
    <div className="floating-badge">
      <div className="floating-badge-glow" />
      <div className="floating-badge-content">
        <span className="floating-badge-dot" />
        <span>Financial Intelligence Platform</span>
      </div>
    </div>
  );
}

// ============================================
// STATS SECTION
// ============================================
function StatsSection() {
  const stats = [
    { value: 5, label: 'Analysis Tools', suffix: '' },
    { value: 3, label: 'Scoring Models', suffix: '' },
    { value: 100, label: 'Data Points', suffix: 'K+' },
    { value: 24, label: 'Updates', suffix: '/7' },
  ];

  return (
    <div className="stats-section">
      <div className="stats-grid">
        {stats.map((stat, index) => (
          <div 
            key={index} 
            className="stat-item"
            style={{ animationDelay: `${0.8 + index * 0.1}s` }}
          >
            <div className="stat-value">
              <AnimatedCounter value={stat.value} suffix={stat.suffix} />
            </div>
            <div className="stat-label">{stat.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================
// TRUSTED BY SECTION
// ============================================
function TrustedSection() {
  return (
    <div className="trusted-section">
      <p className="trusted-label">Powered by industry-leading data sources</p>
      <div className="trusted-logos">
        {['Yahoo Finance', 'SEC EDGAR', 'StockTwits', 'News APIs'].map((source, i) => (
          <div key={i} className="trusted-logo" style={{ animationDelay: `${1 + i * 0.1}s` }}>
            {source}
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================
// MAIN PAGE COMPONENT
// ============================================
export default function Home() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) return null;

  return (
    <>
      {/* Background layers */}
      <div className="page-background">
        <GradientOrbs />
        <div className="grid-overlay" />
        <div className="noise-overlay" />
      </div>
      
      <ParticleNetwork />

      <div className="home-container">
        {/* Hero Section */}
        <section className="hero-section">
          <FloatingBadge />
          
          <h1 className="hero-title">
            <span className="hero-title-line">
              <TextReveal delay={0.3}>Make Smarter</TextReveal>
            </span>
            <span className="hero-title-line gradient-text">
              <TextReveal delay={0.5}>Investment Decisions</TextReveal>
            </span>
          </h1>
          
          <p className="hero-subtitle">
            <TextReveal delay={0.7}>
              Professional-grade financial intelligence combining real-time sentiment analysis, academic scoring models, and insider activity tracking.
            </TextReveal>
          </p>

          <div className="hero-cta">
            <MagneticButton href="/search" variant="primary">
              <span>Start Analyzing</span>
              {icons.arrow}
            </MagneticButton>
            <MagneticButton href="/sentiment" variant="secondary">
              <span>View Sentiment</span>
            </MagneticButton>
          </div>

          {/* Scroll indicator */}
          <div className="scroll-indicator">
            <div className="scroll-indicator-track">
              <div className="scroll-indicator-thumb" />
            </div>
            <span>Scroll to explore</span>
          </div>
        </section>

        {/* Features Section */}
        <section className="features-section">
          <div className="section-header">
            <span className="section-label">Tools</span>
            <h2 className="section-title">Everything you need to invest with confidence</h2>
          </div>

          <div className="features-grid">
            {features.map((feature, index) => (
              <FeatureCard key={feature.id} feature={feature} index={index} />
            ))}
          </div>
        </section>

        {/* Stats Section */}
        <StatsSection />

        {/* Trusted Section */}
        <TrustedSection />

        {/* CTA Section */}
        <section className="cta-section">
          <div className="cta-card">
            <div className="cta-glow" />
            <h2>Ready to elevate your investment strategy?</h2>
            <p>Join thousands of investors using Caveray for smarter decisions.</p>
            <MagneticButton href="/search" variant="primary">
              <span>Get Started Free</span>
              {icons.arrow}
            </MagneticButton>
          </div>
        </section>
      </div>
    </>
  );
}
