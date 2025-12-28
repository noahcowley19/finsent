'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import { MobileMenu } from '@/components/layout/MobileMenu';

// =============================================================================
// TYPES
// =============================================================================

export interface NavLink {
  label: string;
  href: string;
  requiresAuth?: boolean;
  requiresPro?: boolean;
  icon?: React.ReactNode;
}

export interface User {
  id: string;
  email: string;
  name?: string;
  image?: string;
  tier: 'free' | 'pro';
}

export interface NavbarProps {
  transparent?: boolean;
  user?: User | null;
  onSignIn?: () => void;
  onSignUp?: () => void;
  onSignOut?: () => void;
}

// =============================================================================
// NAVIGATION CONFIG
// =============================================================================

export const navLinks: NavLink[] = [
  { label: 'Dashboard', href: '/dashboard' },
  { label: 'Search', href: '/search' },
  { label: 'Sentiment', href: '/sentiment' },
  { label: 'Screener', href: '/screener' },
  { label: 'Movers', href: '/movers' },
  { label: 'Sectors', href: '/sectors' },
  { label: 'Technicals', href: '/technicals' },
  { label: 'Compare', href: '/compare' },
  { label: 'Dividends', href: '/dividends' },
  { label: 'Earnings', href: '/earnings' },
  { label: 'Portfolio', href: '/portfolio' },
  { label: 'Watchlist', href: '/watchlist' },
  { label: 'Journal', href: '/journal' },
  { label: 'Quant Lab', href: '/quant-lab' },
];

const featuredLinks: NavLink[] = [
  { label: 'Search', href: '/search' },
  { label: 'Dashboard', href: '/dashboard' },
  { label: 'Pricing', href: '/pricing' },
];

// =============================================================================
// LOGO COMPONENT
// =============================================================================

const Logo: React.FC<{ scrolled?: boolean; transparent?: boolean }> = () => {
  return (
    <Link href="/" className="flex items-center gap-2.5 group">
      <div className="relative w-9 h-9 transition-transform duration-200 group-hover:scale-105">
        <Image
          src="/logo.png"
          alt="Caveray"
          width={36}
          height={36}
          className="object-contain"
          priority
        />
      </div>
      <span className="font-heading font-semibold text-xl text-navy-900 transition-colors duration-200">
        Caveray
      </span>
    </Link>
  );
};

// =============================================================================
// USER MENU COMPONENT
// =============================================================================

const UserMenu: React.FC<{
  user: User;
  onSignOut?: () => void;
}> = ({ user, onSignOut }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 p-1.5 rounded-full transition-all duration-200 hover:bg-navy-100/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy-500"
      >
        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-terra-400 to-terra-600 flex items-center justify-center text-white text-sm font-medium shadow-sm">
          {user.name?.charAt(0) || user.email.charAt(0).toUpperCase()}
        </div>
        <svg className={`w-4 h-4 text-navy-500 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-dropdown" onClick={() => setIsOpen(false)} />
          <div className="absolute right-0 mt-2 w-64 bg-white/95 backdrop-blur-xl rounded-2xl shadow-xl border border-navy-100/50 z-dropdown animate-fade-in-down overflow-hidden">
            <div className="px-4 py-4 bg-cream-50/50">
              <p className="text-body-md font-semibold text-navy-900 truncate">{user.name || 'User'}</p>
              <p className="text-body-sm text-navy-500 truncate">{user.email}</p>
              {user.tier === 'pro' && (
                <span className="inline-flex items-center mt-2 px-2.5 py-1 rounded-full text-caption font-semibold bg-gradient-to-r from-terra-500 to-pink-500 text-white">
                  Pro Member
                </span>
              )}
            </div>
            <div className="py-2">
              <Link href="/settings" onClick={() => setIsOpen(false)} className="flex items-center gap-3 px-4 py-2.5 text-body-sm text-navy-700 hover:bg-cream-50 transition-colors">
                <svg className="w-5 h-5 text-navy-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
                Account Settings
              </Link>
              {user.tier === 'free' && (
                <Link href="/pricing" onClick={() => setIsOpen(false)} className="flex items-center gap-3 px-4 py-2.5 text-body-sm text-terra-600 hover:bg-cream-50 transition-colors">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                  </svg>
                  Upgrade to Pro
                </Link>
              )}
            </div>
            <div className="border-t border-navy-100/50 py-2">
              <button onClick={() => { setIsOpen(false); onSignOut?.(); }} className="flex items-center gap-3 w-full px-4 py-2.5 text-body-sm text-navy-500 hover:bg-cream-50 transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                </svg>
                Sign Out
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

// =============================================================================
// NAVBAR COMPONENT
// =============================================================================

export const Navbar: React.FC<NavbarProps> = ({
  transparent = false,
  user = null,
  onSignIn,
  onSignUp,
  onSignOut,
}) => {
  const [scrolled, setScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const pathname = usePathname();

  const handleScroll = useCallback(() => {
    setScrolled(window.scrollY > 20);
  }, []);

  useEffect(() => {
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);

  const visibleLinks = navLinks.filter((link) => {
    if (link.requiresAuth && !user) return false;
    if (link.requiresPro && user?.tier !== 'pro') return false;
    return true;
  });

  return (
    <>
      <header className={`fixed top-0 left-0 right-0 z-fixed transition-all duration-300 ease-out ${scrolled ? 'bg-white/80 backdrop-blur-xl shadow-sm border-b border-navy-100/30' : transparent ? 'bg-transparent' : 'bg-cream-50/80 backdrop-blur-md'}`}>
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className={`flex items-center justify-between transition-all duration-300 ${scrolled ? 'h-16' : 'h-20'}`}>
            <Logo scrolled={scrolled} transparent={transparent} />

            <div className="hidden lg:flex items-center gap-1">
              {featuredLinks.map((link) => {
                const isActive = pathname === link.href || (link.href !== '/' && pathname.startsWith(link.href));
                return (
                  <Link key={link.href} href={link.href} className={`relative px-4 py-2 rounded-full text-body-sm font-medium transition-all duration-200 ${isActive ? 'text-navy-900 bg-navy-100/50' : 'text-navy-600 hover:text-navy-900 hover:bg-navy-50'}`}>
                    {link.label}
                  </Link>
                );
              })}
            </div>

            <div className="flex items-center gap-3">
              {user ? (
                <UserMenu user={user} onSignOut={onSignOut} />
              ) : (
                <div className="hidden sm:flex items-center gap-3">
                  <button onClick={onSignIn} className="px-4 py-2 text-body-sm font-medium rounded-full text-navy-600 hover:text-navy-900 transition-colors duration-200">
                    Log in
                  </button>
                  <button onClick={onSignUp} className="px-5 py-2.5 text-body-sm font-semibold rounded-full bg-navy-900 text-white hover:bg-navy-800 transition-all duration-200 shadow-sm hover:shadow-md">
                    Get started
                  </button>
                </div>
              )}

              <button onClick={() => setMobileMenuOpen(true)} className="lg:hidden p-2.5 rounded-full text-navy-600 hover:text-navy-900 hover:bg-navy-100/50 transition-colors duration-200" aria-label="Open menu">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                </svg>
              </button>
            </div>
          </div>
        </nav>
      </header>

      <MobileMenu isOpen={mobileMenuOpen} onClose={() => setMobileMenuOpen(false)} links={visibleLinks} user={user} onSignIn={onSignIn} onSignUp={onSignUp} onSignOut={onSignOut} />

      <div className={`${scrolled ? 'h-16' : 'h-20'} transition-all duration-300`} />
    </>
  );
};

export default Navbar;
