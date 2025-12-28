'use client';

import React, { useEffect, useRef } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { NavLink, User } from './Navbar';

// =============================================================================
// TYPES
// =============================================================================

export interface MobileMenuProps {
  isOpen: boolean;
  onClose: () => void;
  links: NavLink[];
  user?: User | null;
  onSignIn?: () => void;
  onSignUp?: () => void;
  onSignOut?: () => void;
}

// =============================================================================
// MOBILE MENU COMPONENT
// =============================================================================

export const MobileMenu: React.FC<MobileMenuProps> = ({
  isOpen,
  onClose,
  links,
  user,
  onSignIn,
  onSignUp,
  onSignOut,
}) => {
  const menuRef = useRef<HTMLDivElement>(null);

  // Lock body scroll when menu is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = '';
    }
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  // Close on escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
    }
    return () => document.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-ink-950/20 backdrop-blur-sm z-modal-backdrop animate-fade-in"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Menu Panel */}
      <div
        ref={menuRef}
        className="fixed inset-y-0 right-0 w-full max-w-sm bg-white shadow-2xl z-modal animate-slide-in-right"
        role="dialog"
        aria-modal="true"
      >
        {/* Header */}
        <div className="flex items-center justify-between h-16 px-6 border-b border-ink-100">
          <Image
            src="/caveray-wordmark.png"
            alt="Caveray"
            width={100}
            height={24}
            className="h-6 w-auto object-contain"
          />
          <button
            onClick={onClose}
            className="p-2 -mr-2 rounded-lg text-ink-500 hover:text-ink-700 hover:bg-ink-100 transition-colors"
            aria-label="Close menu"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Navigation Links */}
        <nav className="flex-1 overflow-y-auto py-4">
          <div className="px-3 space-y-1">
            {links.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                onClick={onClose}
                className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-body-md text-ink-700 hover:text-ink-900 hover:bg-ink-50 transition-colors"
              >
                {link.icon && (
                  <span className="w-5 h-5 text-ink-400">{link.icon}</span>
                )}
                {link.label}
              </Link>
            ))}
          </div>
        </nav>

        {/* Footer */}
        <div className="p-6 border-t border-ink-100">
          {user ? (
            <div className="space-y-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-gradient-to-br from-ink-600 to-ink-800 flex items-center justify-center text-white font-medium">
                  {user.name?.charAt(0) || user.email.charAt(0).toUpperCase()}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-body-sm font-medium text-ink-900 truncate">
                    {user.name || 'User'}
                  </p>
                  <p className="text-body-xs text-ink-500 truncate">{user.email}</p>
                </div>
                {user.tier === 'pro' && (
                  <span className="px-2 py-0.5 rounded-full text-body-xs font-medium bg-accent/10 text-accent">
                    Pro
                  </span>
                )}
              </div>
              <div className="flex gap-2">
                <Link
                  href="/settings"
                  onClick={onClose}
                  className="flex-1 px-4 py-2.5 text-center text-body-sm font-medium text-ink-700 bg-ink-100 hover:bg-ink-200 rounded-lg transition-colors"
                >
                  Settings
                </Link>
                <button
                  onClick={() => { onClose(); onSignOut?.(); }}
                  className="flex-1 px-4 py-2.5 text-body-sm font-medium text-ink-500 border border-ink-200 hover:bg-ink-50 rounded-lg transition-colors"
                >
                  Sign out
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <button
                onClick={() => { onClose(); onSignUp?.(); }}
                className="w-full px-4 py-2.5 text-body-sm font-medium text-white bg-ink-900 hover:bg-ink-800 rounded-lg transition-colors"
              >
                Get started
              </button>
              <button
                onClick={() => { onClose(); onSignIn?.(); }}
                className="w-full px-4 py-2.5 text-body-sm font-medium text-ink-700 border border-ink-200 hover:bg-ink-50 rounded-lg transition-colors"
              >
                Log in
              </button>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default MobileMenu;
