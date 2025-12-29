'use client';

// =============================================================================
// FINAL CTA DARK - Dark Theme Call to Action
// =============================================================================
// Final CTA section for the dark-themed Financial OS landing page
//
// Location: frontend/components/landing/FinalCTADark.tsx
// =============================================================================

import React from 'react';
import Link from 'next/link';
import { motion } from 'framer-motion';

export const FinalCTADark: React.FC = () => {
    return (
        <section className="py-24 lg:py-32 bg-obsidian-950 relative overflow-hidden">
            {/* Background effects */}
            <div className="absolute inset-0">
                <div
                    className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px]"
                    style={{
                        background: 'radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%)',
                    }}
                />
            </div>

            <div className="relative z-10 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                >
                    <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold text-white mb-6 tracking-tight">
                        Ready to upgrade your
                        <br />
                        <span className="text-gradient">investment workflow?</span>
                    </h2>
                    <p className="text-lg text-obsidian-400 max-w-2xl mx-auto mb-10">
                        Join thousands of investors using Caveray to make smarter, data-driven decisions.
                        Start free today.
                    </p>

                    <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
                        <Link
                            href="/signup"
                            className="group flex items-center gap-2 px-8 py-4 bg-electric-500 text-white font-semibold rounded-xl hover:bg-electric-600 transition-all duration-200 hover:-translate-y-1 hover:shadow-glow-lg"
                        >
                            Start Building Free
                            <svg className="w-5 h-5 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                            </svg>
                        </Link>
                        <Link
                            href="/dashboard"
                            className="flex items-center gap-2 px-8 py-4 border border-obsidian-700 text-obsidian-300 font-medium rounded-xl hover:border-obsidian-600 hover:text-white transition-all duration-200"
                        >
                            View Live Demo
                        </Link>
                    </div>

                    <p className="text-sm text-obsidian-500 mt-8">
                        No credit card required • 14-day free trial on Pro
                    </p>
                </motion.div>
            </div>
        </section>
    );
};

export default FinalCTADark;
