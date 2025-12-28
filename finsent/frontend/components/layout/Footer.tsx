'use client';

import React from 'react';
import Link from 'next/link';
import Image from 'next/image';
import { FaXTwitter, FaGithub, FaLinkedinIn } from 'react-icons/fa6';

// =============================================================================
// TYPES
// =============================================================================

export type FooterVariant = 'default' | 'minimal';

export interface FooterProps {
  variant?: FooterVariant;
}

// =============================================================================
// FOOTER LINKS CONFIG
// =============================================================================

const footerLinks = {
  product: {
    title: 'Product',
    links: [
      { label: 'Stock Search', href: '/search' },
      { label: 'Sentiment Analysis', href: '/sentiment' },
      { label: 'Financial Analyzer', href: '/financials' },
      { label: 'Insider Trading', href: '/insider' },
      { label: 'Portfolio Tracker', href: '/portfolio' },
      { label: 'Quant Lab', href: '/quant-lab' },
    ],
  },
  company: {
    title: 'Company',
    links: [
      { label: 'About', href: '/about' },
      { label: 'Pricing', href: '/pricing' },
      { label: 'Blog', href: '/blog' },
      { label: 'Changelog', href: '/changelog' },
    ],
  },
  resources: {
    title: 'Resources',
    links: [
      { label: 'Documentation', href: '/docs' },
      { label: 'API Reference', href: '/api' },
      { label: 'Help Center', href: '/help' },
      { label: 'Contact', href: '/contact' },
    ],
  },
  legal: {
    title: 'Legal',
    links: [
      { label: 'Privacy Policy', href: '/privacy' },
      { label: 'Terms of Service', href: '/terms' },
      { label: 'Cookie Policy', href: '/cookies' },
    ],
  },
};

const socialLinks = [
  {
    label: 'Twitter',
    href: 'https://twitter.com/caveray',
    icon: FaXTwitter,
  },
  {
    label: 'GitHub',
    href: 'https://github.com/caveray',
    icon: FaGithub,
  },
  {
    label: 'LinkedIn',
    href: 'https://linkedin.com/company/caveray',
    icon: FaLinkedinIn,
  },
];

// =============================================================================
// COMPONENT
// =============================================================================

export const Footer: React.FC<FooterProps> = ({ variant = 'default' }) => {
  const currentYear = new Date().getFullYear();

  if (variant === 'minimal') {
    return (
      <footer className="bg-cream-50 border-t border-navy-100/30">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            {/* Logo */}
            <Link href="/" className="flex items-center gap-2">
              <div className="relative w-7 h-7">
                <Image src="/logo.png" alt="Caveray" fill className="object-contain" />
              </div>
              <span className="font-heading font-semibold text-navy-900">Caveray</span>
            </Link>

            {/* Copyright */}
            <p className="text-body-sm text-navy-400">
              © {currentYear} Caveray. All rights reserved.
            </p>

            {/* Links */}
            <div className="flex items-center gap-6">
              <Link href="/privacy" className="text-body-sm text-navy-400 hover:text-navy-700 transition-colors">
                Privacy
              </Link>
              <Link href="/terms" className="text-body-sm text-navy-400 hover:text-navy-700 transition-colors">
                Terms
              </Link>
            </div>
          </div>
        </div>
      </footer>
    );
  }

  return (
    <footer className="bg-navy-900 text-white">
      {/* Main footer */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16 lg:py-20">
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-10 lg:gap-12">
          {/* Brand column */}
          <div className="col-span-2 md:col-span-4 lg:col-span-1 mb-8 lg:mb-0">
            <Link href="/" className="flex items-center gap-2.5 mb-5">
              <div className="relative w-9 h-9">
                <Image src="/logo.png" alt="Caveray" fill className="object-contain brightness-0 invert" />
              </div>
              <span className="font-heading font-semibold text-xl">Caveray</span>
            </Link>
            <p className="text-body-sm text-white/50 mb-6 max-w-xs leading-relaxed">
              Financial intelligence platform powered by AI. Make smarter investment decisions with comprehensive market analysis.
            </p>

            {/* Social links */}
            <div className="flex items-center gap-3">
              {socialLinks.map((social) => {
                const Icon = social.icon;
                return (
                  <a
                    key={social.label}
                    href={social.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="
                      w-10 h-10 rounded-full
                      bg-white/5 hover:bg-white/10
                      flex items-center justify-center
                      text-white/50 hover:text-white
                      transition-all duration-200
                    "
                    aria-label={social.label}
                  >
                    <Icon className="w-4 h-4" />
                  </a>
                );
              })}
            </div>
          </div>

          {/* Link columns */}
          {Object.values(footerLinks).map((section) => (
            <div key={section.title}>
              <h4 className="font-heading font-semibold text-white mb-5">
                {section.title}
              </h4>
              <ul className="space-y-3">
                {section.links.map((link) => (
                  <li key={link.href}>
                    <Link
                      href={link.href}
                      className="text-body-sm text-white/50 hover:text-white transition-colors"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom bar */}
      <div className="border-t border-white/10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
            <p className="text-body-sm text-white/40">
              © {currentYear} Caveray. All rights reserved.
            </p>
            <p className="text-body-sm text-white/30">
              Market data provided for informational purposes only. Not financial advice.
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
