'use client';

// =============================================================================
// COMMAND BAR NAVBAR - Terminal-Inspired Navigation
// =============================================================================
// A dark, terminal-style navigation with grouped links and glowing active states
//
// Location: frontend/components/layout/CommandBarNavbar.tsx
// =============================================================================

import React, { useState, useEffect, useCallback } from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { usePathname } from 'next/navigation';
import { motion, AnimatePresence } from 'framer-motion';

// =============================================================================
// TYPES
// =============================================================================

interface NavLink {
    label: string;
    href: string;
    description: string;
    icon: React.ReactNode;
}

interface NavGroup {
    id: string;
    label: string;
    links: NavLink[];
}

// =============================================================================
// ICONS
// =============================================================================

const MacroIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
);

const InsiderIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
    </svg>
);

const ScreenerIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
    </svg>
);

const QuantIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
    </svg>
);

const PortfolioIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
    </svg>
);

// =============================================================================
// NAVIGATION CONFIG
// =============================================================================

const navGroups: NavGroup[] = [
    {
        id: 'market-state',
        label: 'Market State',
        links: [
            {
                label: 'Macro Command',
                href: '/economic',
                description: 'Yield curves, recession probability, Fed signals',
                icon: <MacroIcon />,
            },
            {
                label: 'Insider Radar',
                href: '/insider',
                description: 'Track smart money moves in real-time',
                icon: <InsiderIcon />,
            },
        ],
    },
    {
        id: 'analysis',
        label: 'Analysis',
        links: [
            {
                label: 'Stock Screener',
                href: '/screener',
                description: 'Filter 50K+ stocks by sentiment & fundamentals',
                icon: <ScreenerIcon />,
            },
            {
                label: 'Quant Lab',
                href: '/quant-lab',
                description: 'Backtest strategies with Python-powered analytics',
                icon: <QuantIcon />,
            },
        ],
    },
];

const portfolioLink: NavLink = {
    label: 'Portfolio',
    href: '/portfolio',
    description: 'Track your holdings with advanced risk analytics',
    icon: <PortfolioIcon />,
};

// =============================================================================
// DROPDOWN COMPONENT
// =============================================================================

const NavDropdown: React.FC<{
    group: NavGroup;
    isOpen: boolean;
    onToggle: () => void;
    onClose: () => void;
}> = ({ group, isOpen, onToggle, onClose }) => {
    const pathname = usePathname();
    const isActive = group.links.some(link => pathname === link.href || pathname.startsWith(link.href + '/'));

    return (
        <div className="relative">
            <button
                onClick={onToggle}
                className={`
          relative flex items-center gap-1.5 px-3 py-2 rounded-lg
          text-sm font-medium transition-colors duration-150
          ${isActive ? 'text-white' : 'text-obsidian-400 hover:text-white'}
        `}
            >
                {group.label}
                <svg
                    className={`w-3.5 h-3.5 transition-transform duration-150 ${isOpen ? 'rotate-180' : ''}`}
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
                {isActive && (
                    <motion.div
                        layoutId="navbar-active-indicator"
                        className="absolute -bottom-1 left-0 right-0 h-0.5 bg-gradient-to-r from-electric-400 to-electric-600 rounded-full"
                        style={{ boxShadow: '0 0 12px rgba(59, 130, 246, 0.7)' }}
                    />
                )}
            </button>

            <AnimatePresence>
                {isOpen && (
                    <>
                        <div className="fixed inset-0 z-dropdown" onClick={onClose} />
                        <motion.div
                            initial={{ opacity: 0, y: -8 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -8 }}
                            transition={{ duration: 0.15 }}
                            className="absolute top-full left-0 mt-2 w-72 z-dropdown"
                        >
                            <div className="bg-obsidian-900/95 backdrop-blur-xl border border-obsidian-800 rounded-2xl shadow-2xl overflow-hidden">
                                {group.links.map((link) => {
                                    const linkActive = pathname === link.href || pathname.startsWith(link.href + '/');
                                    return (
                                        <Link
                                            key={link.href}
                                            href={link.href}
                                            onClick={onClose}
                                            className={`
                        flex items-start gap-3 p-4 transition-colors duration-150
                        ${linkActive
                                                    ? 'bg-electric-500/10 border-l-2 border-electric-500'
                                                    : 'hover:bg-obsidian-800/50 border-l-2 border-transparent'
                                                }
                      `}
                                        >
                                            <div className={`p-2 rounded-lg ${linkActive ? 'bg-electric-500/20 text-electric-400' : 'bg-obsidian-800 text-obsidian-400'}`}>
                                                {link.icon}
                                            </div>
                                            <div>
                                                <p className={`text-sm font-semibold ${linkActive ? 'text-white' : 'text-obsidian-200'}`}>
                                                    {link.label}
                                                </p>
                                                <p className="text-xs text-obsidian-500 mt-0.5">
                                                    {link.description}
                                                </p>
                                            </div>
                                        </Link>
                                    );
                                })}
                            </div>
                        </motion.div>
                    </>
                )}
            </AnimatePresence>
        </div>
    );
};

// =============================================================================
// MOBILE DRAWER
// =============================================================================

const MobileDrawer: React.FC<{
    isOpen: boolean;
    onClose: () => void;
}> = ({ isOpen, onClose }) => {
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

    return (
        <AnimatePresence>
            {isOpen && (
                <>
                    {/* Backdrop */}
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="fixed inset-0 bg-obsidian-950/80 backdrop-blur-sm z-modal-backdrop"
                        onClick={onClose}
                    />

                    {/* Drawer */}
                    <motion.div
                        initial={{ y: '-100%' }}
                        animate={{ y: 0 }}
                        exit={{ y: '-100%' }}
                        transition={{ type: 'spring', damping: 25, stiffness: 300 }}
                        className="fixed top-0 left-0 right-0 bg-obsidian-950/98 backdrop-blur-3xl border-b border-electric-500/20 z-modal max-h-[85vh] overflow-y-auto"
                    >
                        {/* Header */}
                        <div className="flex items-center justify-between px-6 py-4 border-b border-obsidian-800">
                            <Link href="/" onClick={onClose} className="flex items-center gap-2">
                                <Image
                                    src="/caveray-wordmark.png"
                                    alt="Caveray"
                                    width={120}
                                    height={28}
                                    className="h-7 w-auto object-contain brightness-0 invert"
                                />
                            </Link>
                            <button
                                onClick={onClose}
                                className="p-2 rounded-lg text-obsidian-400 hover:text-white hover:bg-obsidian-800 transition-colors"
                            >
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                                </svg>
                            </button>
                        </div>

                        {/* Navigation */}
                        <nav className="px-6 py-8 space-y-8">
                            {navGroups.map((group) => (
                                <div key={group.id} className="space-y-3">
                                    <p className="text-xs font-medium text-obsidian-500 uppercase tracking-wider">
                                        {group.label}
                                    </p>
                                    <div className="space-y-2">
                                        {group.links.map((link) => {
                                            const isActive = pathname === link.href || pathname.startsWith(link.href + '/');
                                            return (
                                                <Link
                                                    key={link.href}
                                                    href={link.href}
                                                    onClick={onClose}
                                                    className={`
                            flex items-center gap-4 p-4 rounded-2xl transition-all duration-150
                            ${isActive
                                                            ? 'bg-electric-500/10 border border-electric-500/30'
                                                            : 'bg-obsidian-900/50 border border-obsidian-800 hover:border-electric-500/30'
                                                        }
                          `}
                                                >
                                                    <div className={`p-3 rounded-xl ${isActive ? 'bg-electric-500/20 text-electric-400' : 'bg-obsidian-800 text-obsidian-400'}`}>
                                                        {link.icon}
                                                    </div>
                                                    <div>
                                                        <p className={`text-lg font-semibold ${isActive ? 'text-white' : 'text-obsidian-200'}`}>
                                                            {link.label}
                                                        </p>
                                                        <p className="text-sm text-obsidian-500">
                                                            {link.description}
                                                        </p>
                                                    </div>
                                                </Link>
                                            );
                                        })}
                                    </div>
                                </div>
                            ))}

                            {/* Portfolio Link */}
                            <div className="space-y-3">
                                <p className="text-xs font-medium text-obsidian-500 uppercase tracking-wider">
                                    Your Account
                                </p>
                                <Link
                                    href={portfolioLink.href}
                                    onClick={onClose}
                                    className={`
                    flex items-center gap-4 p-4 rounded-2xl transition-all duration-150
                    ${pathname === portfolioLink.href
                                            ? 'bg-electric-500/10 border border-electric-500/30'
                                            : 'bg-obsidian-900/50 border border-obsidian-800 hover:border-electric-500/30'
                                        }
                  `}
                                >
                                    <div className={`p-3 rounded-xl ${pathname === portfolioLink.href ? 'bg-electric-500/20 text-electric-400' : 'bg-obsidian-800 text-obsidian-400'}`}>
                                        {portfolioLink.icon}
                                    </div>
                                    <div>
                                        <p className={`text-lg font-semibold ${pathname === portfolioLink.href ? 'text-white' : 'text-obsidian-200'}`}>
                                            {portfolioLink.label}
                                        </p>
                                        <p className="text-sm text-obsidian-500">
                                            {portfolioLink.description}
                                        </p>
                                    </div>
                                </Link>
                            </div>
                        </nav>

                        {/* CTA */}
                        <div className="px-6 py-6 border-t border-obsidian-800">
                            <Link
                                href="/signup"
                                onClick={onClose}
                                className="flex items-center justify-center gap-2 w-full py-4 bg-electric-500 hover:bg-electric-600 text-white font-semibold rounded-xl transition-colors"
                            >
                                Get Started Free
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                                </svg>
                            </Link>
                        </div>
                    </motion.div>
                </>
            )}
        </AnimatePresence>
    );
};

// =============================================================================
// MAIN COMMAND BAR NAVBAR
// =============================================================================

export const CommandBarNavbar: React.FC = () => {
    const [scrolled, setScrolled] = useState(false);
    const [mobileOpen, setMobileOpen] = useState(false);
    const [openDropdown, setOpenDropdown] = useState<string | null>(null);
    const pathname = usePathname();

    const handleScroll = useCallback(() => {
        setScrolled(window.scrollY > 10);
    }, []);

    useEffect(() => {
        window.addEventListener('scroll', handleScroll, { passive: true });
        handleScroll();
        return () => window.removeEventListener('scroll', handleScroll);
    }, [handleScroll]);

    useEffect(() => {
        setMobileOpen(false);
        setOpenDropdown(null);
    }, [pathname]);

    const portfolioActive = pathname === portfolioLink.href || pathname.startsWith(portfolioLink.href + '/');

    return (
        <>
            <header
                className={`
          fixed top-0 left-0 right-0 z-fixed
          transition-all duration-300 ease-out
          ${scrolled
                        ? 'bg-obsidian-950/95 backdrop-blur-2xl border-b border-electric-500/20'
                        : 'bg-obsidian-950/80 backdrop-blur-xl border-b border-obsidian-800/50'
                    }
        `}
            >
                <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className={`flex items-center justify-between transition-all duration-200 ${scrolled ? 'h-14' : 'h-16'}`}>
                        {/* Logo */}
                        <Link href="/" className="flex items-center gap-2 group">
                            <Image
                                src="/caveray-wordmark.png"
                                alt="Caveray"
                                width={130}
                                height={30}
                                className="h-7 w-auto object-contain brightness-0 invert transition-opacity group-hover:opacity-80"
                                priority
                            />
                        </Link>

                        {/* Desktop Navigation */}
                        <div className="hidden lg:flex items-center gap-1">
                            {navGroups.map((group) => (
                                <NavDropdown
                                    key={group.id}
                                    group={group}
                                    isOpen={openDropdown === group.id}
                                    onToggle={() => setOpenDropdown(openDropdown === group.id ? null : group.id)}
                                    onClose={() => setOpenDropdown(null)}
                                />
                            ))}

                            {/* Portfolio Direct Link */}
                            <Link
                                href={portfolioLink.href}
                                className={`
                  relative flex items-center gap-1.5 px-3 py-2 rounded-lg
                  text-sm font-medium transition-colors duration-150
                  ${portfolioActive ? 'text-white' : 'text-obsidian-400 hover:text-white'}
                `}
                            >
                                {portfolioLink.label}
                                {portfolioActive && (
                                    <motion.div
                                        layoutId="navbar-active-indicator"
                                        className="absolute -bottom-1 left-0 right-0 h-0.5 bg-gradient-to-r from-electric-400 to-electric-600 rounded-full"
                                        style={{ boxShadow: '0 0 12px rgba(59, 130, 246, 0.7)' }}
                                    />
                                )}
                            </Link>
                        </div>

                        {/* Right Side */}
                        <div className="flex items-center gap-3">
                            {/* Desktop Auth */}
                            <div className="hidden lg:flex items-center gap-3">
                                <Link
                                    href="/login"
                                    className="px-4 py-2 text-sm font-medium text-obsidian-400 hover:text-white transition-colors"
                                >
                                    Log in
                                </Link>
                                <Link
                                    href="/signup"
                                    className="px-5 py-2.5 text-sm font-semibold rounded-xl bg-electric-500 text-white hover:bg-electric-600 transition-all duration-150 hover:-translate-y-0.5 hover:shadow-glow-md"
                                >
                                    Get Started
                                </Link>
                            </div>

                            {/* Mobile Menu Button */}
                            <button
                                onClick={() => setMobileOpen(true)}
                                className="lg:hidden p-2 rounded-lg text-obsidian-400 hover:text-white hover:bg-obsidian-800 transition-colors"
                                aria-label="Open menu"
                            >
                                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 6h16M4 12h16M4 18h16" />
                                </svg>
                            </button>
                        </div>
                    </div>
                </nav>
            </header>

            {/* Mobile Drawer */}
            <MobileDrawer isOpen={mobileOpen} onClose={() => setMobileOpen(false)} />

            {/* Spacer */}
            <div className={`${scrolled ? 'h-14' : 'h-16'} transition-all duration-200`} />
        </>
    );
};

export default CommandBarNavbar;
