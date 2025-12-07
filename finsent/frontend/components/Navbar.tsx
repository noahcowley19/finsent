'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useEffect, useState } from 'react';

const navLinks = [
  { href: '/', label: 'Home' },
  { href: '/search', label: 'Stock Search' },
  { href: '/sentiment', label: 'Sentiment' },
  { href: '/financials', label: 'Financials' },
  { href: '/insider', label: 'Insider' },
  { href: '/portfolio', label: 'Portfolio' },
];

export default function Navbar() {
  const pathname = usePathname();
  const [scrolled, setScrolled] = useState(false);
  const [hoveredLink, setHoveredLink] = useState<string | null>(null);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <nav 
      className="navbar"
      style={{
        background: scrolled 
          ? 'rgba(3, 5, 8, 0.95)' 
          : 'rgba(3, 5, 8, 0.7)',
        borderBottomColor: scrolled 
          ? 'rgba(255, 255, 255, 0.08)' 
          : 'rgba(255, 255, 255, 0.04)',
      }}
    >
      {/* Logo */}
      <Link href="/" className="nav-logo group">
        <span className="flex items-center gap-3">
          {/* Animated Logo Icon */}
          <span className="relative w-10 h-10 flex items-center justify-center">
            <svg 
              viewBox="0 0 40 40" 
              fill="none" 
              className="w-full h-full"
              style={{
                filter: 'drop-shadow(0 0 8px rgba(0, 212, 170, 0.4))',
              }}
            >
              <defs>
                <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stopColor="#00d4aa" />
                  <stop offset="100%" stopColor="#00a3ff" />
                </linearGradient>
              </defs>
              {/* Hexagon shape */}
              <path
                d="M20 2L36 11V29L20 38L4 29V11L20 2Z"
                fill="url(#logoGradient)"
                fillOpacity="0.15"
                stroke="url(#logoGradient)"
                strokeWidth="1.5"
              />
              {/* Inner triangle/ray pattern */}
              <path
                d="M20 8L28 22H12L20 8Z"
                fill="url(#logoGradient)"
                fillOpacity="0.5"
              />
              <path
                d="M20 14L24 22H16L20 14Z"
                fill="url(#logoGradient)"
              />
              {/* Center dot */}
              <circle cx="20" cy="26" r="2" fill="url(#logoGradient)" />
            </svg>
            {/* Pulse ring */}
            <span 
              className="absolute inset-0 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-500"
              style={{
                background: 'radial-gradient(circle, rgba(0, 212, 170, 0.2) 0%, transparent 70%)',
                animation: 'pulse 2s ease-in-out infinite',
              }}
            />
          </span>
          
          {/* Logo Text */}
          <span 
            className="text-xl font-bold tracking-tight"
            style={{
              background: 'linear-gradient(135deg, #f8fafc 0%, #94a3b8 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}
          >
            Caveray
          </span>
        </span>
      </Link>

      {/* Navigation Links */}
      <div className="nav-links">
        {navLinks.map((link) => {
          const isActive = pathname === link.href;
          const isHovered = hoveredLink === link.href;
          
          return (
            <Link
              key={link.href}
              href={link.href}
              className={`nav-link ${isActive ? 'active' : ''}`}
              onMouseEnter={() => setHoveredLink(link.href)}
              onMouseLeave={() => setHoveredLink(null)}
              style={{
                color: isActive 
                  ? 'var(--accent)' 
                  : isHovered 
                    ? 'var(--text-primary)' 
                    : 'var(--text-secondary)',
              }}
            >
              <span className="relative z-10">{link.label}</span>
            </Link>
          );
        })}
      </div>

      {/* Mobile Menu Button (placeholder) */}
      <button 
        className="md:hidden p-2 rounded-lg hover:bg-white/5 transition-colors"
        aria-label="Menu"
      >
        <svg 
          width="24" 
          height="24" 
          viewBox="0 0 24 24" 
          fill="none" 
          stroke="currentColor" 
          strokeWidth="2"
          strokeLinecap="round"
        >
          <line x1="3" y1="6" x2="21" y2="6" />
          <line x1="3" y1="12" x2="21" y2="12" />
          <line x1="3" y1="18" x2="21" y2="18" />
        </svg>
      </button>
    </nav>
  );
}
