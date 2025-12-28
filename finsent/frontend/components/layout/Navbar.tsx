'use client';

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import {
  HiMenu,
  HiX,
  HiChevronDown,
  HiSearch,
  HiChartBar,
  HiCollection,
  HiCog,
  HiLogout,
  HiLightningBolt,
  HiUser
} from 'react-icons/hi';
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
  /** Transparent background (for hero sections) */
  transparent?: boolean;
  /** Current user */
  user?: User | null;
  /** Sign in handler */
  onSignIn?: () => void;
  /** Sign up handler */
  onSignUp?: () => void;
  /** Sign out handler */
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

// Featured links for the main nav (keep it minimal)
const featuredLinks: NavLink[] = [
  { label: 'Search', href: '/search' },
  { label: 'Dashboard', href: '/dashboard' },
  { label: 'Pricing', href: '/pricing' },
];

// =============================================================================
// LOGO COMPONENT
// =============================================================================

const Logo: React.FC<{ scrolled?: boolean; transparent?: boolean }> = ({
  scrolled,
  transparent
}) => {
  return (
    <Link href="/" className="flex items-center gap-2.5 group">
      {/* Logo Image */}
      <div className="relative w-9 h-9 transition-transform duration-200 group-hover:scale-105">
        <Image
          src="/logo.png"
          alt="Caveray"
          fill
          className="object-contain"
          priority
        />
      </div>
      {/* Wordmark */}
      <span className={`
        font-heading font-semibold text-xl
        transition-colors duration-200
        ${transparent && !scrolled ? 'text-navy-900' : 'text-navy-900'}
      `}>
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
  scrolled?: boolean;
  transparent?: boolean;
}> = ({ user, onSignOut }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="
          flex items-center gap-2 p-1.5 rounded-full
          transition-all duration-200
          hover:bg-navy-100/50
          focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-navy-500
        "
      >
        {/* Avatar */}
        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-terra-400 to-terra-600 flex items-center justify-center text-white text-sm font-medium shadow-sm">
          {user.name?.charAt(0) || user.email.charAt(0).toUpperCase()}
        </div>
        {/* Dropdown arrow */}
        <HiChevronDown
          className={`w-4 h-4 text-navy-500 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {/* Dropdown */}
      {isOpen && (
        <>
          {/* Backdrop */}
          <div
            className="fixed inset-0 z-dropdown"
            onClick={() => setIsOpen(false)}
          />

          {/* Menu */}
          <div className="
            absolute right-0 mt-2 w-64 
            bg-white/95 backdrop-blur-xl
            rounded-2xl shadow-xl 
            border border-navy-100/50
            z-dropdown 
            animate-fade-in-down
            overflow-hidden
          ">
            {/* User info */}
            <div className="px-4 py-4 bg-cream-50/50">
              <p className="text-body-md font-semibold text-navy-900 truncate">
                {user.name || 'User'}
              </p>
              <p className="text-body-sm text-navy-500 truncate">
                {user.email}
              </p>
              {user.tier === 'pro' && (
                <span className="inline-flex items-center mt-2 px-2.5 py-1 rounded-full text-caption font-semibold bg-gradient-to-r from-terra-500 to-pink-500 text-white">
                  Pro Member
                </span>
              )}
            </div>

            {/* Menu items */}
            <div className="py-2">
              <Link
                href="/settings"
                onClick={() => setIsOpen(false)}
                className="flex items-center gap-3 px-4 py-2.5 text-body-sm text-navy-700 hover:bg-cream-50 transition-colors"
              >
                <HiCog className="w-5 h-5 text-navy-400" />
                Account Settings
              </Link>

              {user.tier === 'free' && (
                <Link
                  href="/pricing"
                  onClick={() => setIsOpen(false)}
                  className="flex items-center gap-3 px-4 py-2.5 text-body-sm text-terra-600 hover:bg-cream-50 transition-colors"
                >
                  <HiLightningBolt className="w-5 h-5" />
                  Upgrade to Pro
                </Link>
              )}
            </div>

            {/* Sign out */}
            <div className="border-t border-navy-100/50 py-2">
              <button
                onClick={() => {
                  setIsOpen(false);
                  onSignOut?.();
                }}
                className="flex items-center gap-3 w-full px-4 py-2.5 text-body-sm text-navy-500 hover:bg-cream-50 transition-colors"
              >
                <HiLogout className="w-5 h-5" />
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

  // Handle scroll
  const handleScroll = useCallback(() => {
    setScrolled(window.scrollY > 20);
  }, []);

  useEffect(() => {
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
    return () => window.removeEventListener('scroll', handleScroll);
  }, [handleScroll]);

  // Close mobile menu on route change
  useEffect(() => {
    setMobileMenuOpen(false);
  }, [pathname]);

  // Filter visible links based on auth state
  const visibleLinks = navLinks.filter((link) => {
    if (link.requiresAuth && !user) return false;
    if (link.requiresPro && user?.tier !== 'pro') return false;
    return true;
  });

  return (
    <>
      <header
        className={`
          fixed top-0 left-0 right-0 z-fixed
          transition-all duration-300 ease-out
          ${scrolled
            ? 'bg-white/80 backdrop-blur-xl shadow-sm border-b border-navy-100/30'
            : transparent
              ? 'bg-transparent'
              : 'bg-cream-50/80 backdrop-blur-md'
          }
        `}
      >
        <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className={`
            flex items-center justify-between
            transition-all duration-300
            ${scrolled ? 'h-16' : 'h-20'}
          `}>
            {/* Logo */}
            <Logo scrolled={scrolled} transparent={transparent} />

            {/* Desktop Navigation - Minimal links */}
            <div className="hidden lg:flex items-center gap-1">
              {featuredLinks.map((link) => {
                const isActive = pathname === link.href ||
                  (link.href !== '/' && pathname.startsWith(link.href));

                return (
                  <Link
                    key={link.href}
                    href={link.href}
                    className={`
                      relative px-4 py-2 rounded-full
                      text-body-sm font-medium
                      transition-all duration-200
                      ${isActive
                        ? 'text-navy-900 bg-navy-100/50'
                        : 'text-navy-600 hover:text-navy-900 hover:bg-navy-50'
                      }
                    `}
                  >
                    {link.label}
                  </Link>
                );
              })}
            </div>

            {/* Right section */}
            <div className="flex items-center gap-3">
              {user ? (
                <UserMenu
                  user={user}
                  onSignOut={onSignOut}
                  scrolled={scrolled}
                  transparent={transparent}
                />
              ) : (
                <div className="hidden sm:flex items-center gap-3">
                  <button
                    onClick={onSignIn}
                    className="
                      px-4 py-2 text-body-sm font-medium rounded-full
                      text-navy-600 hover:text-navy-900
                      transition-colors duration-200
                    "
                  >
                    Log in
                  </button>
                  <button
                    onClick={onSignUp}
                    className="
                      px-5 py-2.5 text-body-sm font-semibold rounded-full
                      bg-navy-900 text-white
                      hover:bg-navy-800
                      transition-all duration-200
                      shadow-sm hover:shadow-md
                    "
                  >
                    Get started
                  </button>
                </div>
              )}

              {/* Mobile menu button */}
              <button
                onClick={() => setMobileMenuOpen(true)}
                className="
                  lg:hidden p-2.5 rounded-full
                  text-navy-600 hover:text-navy-900
                  hover:bg-navy-100/50
                  transition-colors duration-200
                "
                aria-label="Open menu"
              >
                <HiMenu className="w-6 h-6" />
              </button>
            </div>
          </div>
        </nav>
      </header>

      {/* Mobile menu */}
      <MobileMenu
        isOpen={mobileMenuOpen}
        onClose={() => setMobileMenuOpen(false)}
        links={visibleLinks}
        user={user}
        onSignIn={onSignIn}
        onSignUp={onSignUp}
        onSignOut={onSignOut}
      />

      {/* Spacer to prevent content from going under fixed navbar */}
      <div className={`${scrolled ? 'h-16' : 'h-20'} transition-all duration-300`} />
    </>
  );
};

export default Navbar;
