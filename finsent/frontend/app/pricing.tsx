// =============================================================================
// PRICING PAGE
// =============================================================================
// Standalone pricing page
//
// Location: frontend/app/pricing/page.tsx
//
// =============================================================================

import { Pricing, FAQ } from '@/components/landing';
import { Section, Container } from '@/components/layout';

export const metadata = {
  title: 'Pricing',
  description: 'Simple, transparent pricing for Caveray. Start free and upgrade when you need more.',
};

export default function PricingPage() {
  return (
    <>
      {/* Hero */}
      <Section spacing="lg" background="gradient">
        <Container size="md">
          <div className="text-center">
            <h1 className="font-display text-display-md lg:text-display-lg text-navy-900 mb-4">
              Simple, transparent pricing
            </h1>
            <p className="text-body-lg text-neutral-600 max-w-2xl mx-auto">
              Start free and upgrade when you&apos;re ready. No hidden fees, no surprises.
            </p>
          </div>
        </Container>
      </Section>

      {/* Pricing cards */}
      <Pricing />

      {/* Comparison table */}
      <Section spacing="lg" background="alt">
        <Container>
          <div className="text-center mb-12">
            <h2 className="font-display text-display-sm text-navy-900 mb-4">
              Compare plans
            </h2>
            <p className="text-body-md text-neutral-600">
              See what&apos;s included in each plan
            </p>
          </div>

          <div className="bg-white rounded-2xl border border-border-light overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-cream-50">
                  <th className="text-left py-4 px-6 text-body-sm font-semibold text-navy-900">Feature</th>
                  <th className="text-center py-4 px-6 text-body-sm font-semibold text-navy-900">Free</th>
                  <th className="text-center py-4 px-6 text-body-sm font-semibold text-navy-900">Pro</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border-light">
                {[
                  { feature: 'Stock Search', free: 'Unlimited', pro: 'Unlimited' },
                  { feature: 'Searches per day', free: '10', pro: 'Unlimited' },
                  { feature: 'Sentiment analyses per day', free: '3', pro: 'Unlimited' },
                  { feature: 'Portfolio positions', free: '5', pro: 'Unlimited' },
                  { feature: 'Watchlist stocks', free: '10', pro: 'Unlimited' },
                  { feature: 'Analysis history', free: '7 days', pro: '90 days' },
                  { feature: 'Quant Lab', free: false, pro: true },
                  { feature: 'Export to CSV/PDF', free: false, pro: true },
                  { feature: 'Email alerts', free: false, pro: true },
                  { feature: 'Priority support', free: false, pro: true },
                ].map((row, index) => (
                  <tr key={index}>
                    <td className="py-4 px-6 text-body-sm text-navy-700">{row.feature}</td>
                    <td className="py-4 px-6 text-center">
                      {typeof row.free === 'boolean' ? (
                        row.free ? (
                          <svg className="w-5 h-5 text-success-500 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        ) : (
                          <svg className="w-5 h-5 text-neutral-300 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                          </svg>
                        )
                      ) : (
                        <span className="text-body-sm text-neutral-600">{row.free}</span>
                      )}
                    </td>
                    <td className="py-4 px-6 text-center">
                      {typeof row.pro === 'boolean' ? (
                        row.pro ? (
                          <svg className="w-5 h-5 text-success-500 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                          </svg>
                        ) : (
                          <svg className="w-5 h-5 text-neutral-300 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                          </svg>
                        )
                      ) : (
                        <span className="text-body-sm font-medium text-navy-900">{row.pro}</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Container>
      </Section>

      {/* FAQ */}
      <FAQ />
    </>
  );
}
