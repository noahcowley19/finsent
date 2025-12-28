'use client';

import React, { useEffect } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import type { NavLink, User } from '@/components/layout/Navbar';

export interface MobileMenuProps {
  isOpen: boolean;
  onClose: () => void;
  links: NavLink[];
  user?: User | null;
  onSignIn?: () => void;
  onSignUp?: () => void;
  onSignOut?: () => void;
}

export const MobileMenu: React.FC<MobileMenuProps> = ({
  isOpen,
  onClose,
  links,
  user,
  onSignIn,
  onSignUp,
  onSignOut,
}) => {
  const pathname = usePathname();

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
    <div className="fixed inset-0 z-modal lg:hidden">
      <div className="absolute inset-0 bg-navy-900/60 backdrop-blur-sm animate-fade-in" onClick={onClose} aria-hidden="true" />

      <div className="absolute top-0 right-0 h-full w-full max-w-sm bg-white shadow-2xl animate-slide-in-right">
        <div className="flex items-center justify-between p-5 border-b border-navy-100/50">
          <Link href="/" onClick={onClose} className="flex items-center gap-2.5">
            <div className="relative w-8 h-8">
              <Image src="/logo.png" alt="Caveray" width={32} height={32} className="object-contain" />
            </div>
            <span className="font-heading font-semibold text-xl text-navy-900">Caveray</span>
          </Link>

          <button onClick={onClose} className="p-2.5 rounded-full text-navy-500 hover:text-navy-900 hover:bg-navy-50 transition-colors" aria-label="Close menu">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {user && (
          <div className="p-5 border-b border-navy-100/50 bg-cream-50/50">
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-terra-400 to-pink-500 flex items-center justify-center text-white font-semibold shadow-lg shadow-terra-500/20">
                {user.name?.charAt(0) || user.email.charAt(0).toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-body-md font-semibold text-navy-900 truncate">{user.name || 'User'}</p>
                <p className="text-body-sm text-navy-500 truncate">{user.email}</p>
              </div>
              {user.tier === 'pro' && (
                <span className="px-2.5 py-1 rounded-full text-caption font-semibold bg-gradient-to-r from-terra-500 to-pink-500 text-white">Pro</span>
              )}
            </div>
          </div>
        )}

        <nav className="p-4 overflow-y-auto max-h-[calc(100vh-280px)]">
          <ul className="space-y-1">
            {links.map((link, index) => {
              const isActive = pathname === link.href || (link.href !== '/' && pathname.startsWith(link.href));
              return (
                <li key={link.href} className="animate-fade-in-up" style={{ animationDelay: `${index * 30}ms` }}>
                  <Link href={link.href} onClick={onClose} className={`flex items-center gap-3 px-4 py-3 rounded-xl text-body-md font-medium transition-all duration-200 ${isActive ? 'bg-navy-900 text-white' : 'text-navy-700 hover:bg-navy-50'}`}>
                    <span className="flex-1">{link.label}</span>
                    {link.requiresPro && (
                      <span className="px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider bg-gradient-to-r from-terra-500 to-pink-500 text-white rounded-full">Pro</span>
                    )}
                  </Link>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="absolute bottom-0 left-0 right-0 p-5 border-t border-navy-100/50 bg-white">
          {user ? (
            <div className="space-y-2">
              <Link href="/settings" onClick={onClose} className="flex items-center justify-center gap-2 w-full px-4 py-3 rounded-xl border border-navy-200 text-navy-700 font-medium hover:bg-navy-50 transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                Account Settings
              </Link>
              {user.tier === 'free' && (
                <Link href="/pricing" onClick={onClose} className="flex items-center justify-center gap-2 w-full px-4 py-3 rounded-xl bg-gradient-to-r from-terra-500 to-pink-500 text-white font-semibold hover:opacity-90 transition-opacity shadow-lg shadow-terra-500/20">
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>
                  Upgrade to Pro
                </Link>
              )}
              <button onClick={() => { onClose(); onSignOut?.(); }} className="flex items-center justify-center gap-2 w-full px-4 py-3 rounded-xl text-navy-500 font-medium hover:bg-navy-50 transition-colors">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" /></svg>
                Sign Out
              </button>
            </div>
          ) : (
            <div className="space-y-3">
              <button onClick={() => { onClose(); onSignUp?.(); }} className="w-full px-4 py-3.5 rounded-xl bg-navy-900 text-white font-semibold hover:bg-navy-800 transition-colors shadow-lg shadow-navy-900/20">
                Get Started
              </button>
              <button onClick={() => { onClose(); onSignIn?.(); }} className="w-full px-4 py-3.5 rounded-xl border border-navy-200 text-navy-700 font-medium hover:bg-navy-50 transition-colors">
                Log In
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MobileMenu;
