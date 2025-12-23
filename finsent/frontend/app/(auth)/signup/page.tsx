'use client';

// =============================================================================
// SIGN UP PAGE
// =============================================================================

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import { signUpSchema, calculatePasswordStrength, type SignUpInput } from '@/lib/validations/auth';
import { ZodError } from 'zod';

export default function SignUpPage() {
  const router = useRouter();
  const { isAuthenticated, isLoading: authLoading } = useAuth();
  
  const [formData, setFormData] = useState<SignUpInput>({
    name: '',
    email: '',
    password: '',
    confirmPassword: '',
    acceptTerms: false,
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [generalError, setGeneralError] = useState('');
  const [passwordStrength, setPasswordStrength] = useState({ score: 0, label: 'Very Weak', suggestions: [] as string[] });

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated && !authLoading) {
      router.push('/');
    }
  }, [isAuthenticated, authLoading, router]);

  // Update password strength on password change
  useEffect(() => {
    if (formData.password) {
      setPasswordStrength(calculatePasswordStrength(formData.password));
    } else {
      setPasswordStrength({ score: 0, label: 'Very Weak', suggestions: [] });
    }
  }, [formData.password]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value,
    }));
    // Clear field error on change
    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: '' }));
    }
    setGeneralError('');
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrors({});
    setGeneralError('');
    setIsSubmitting(true);

    try {
      // Validate form data
      const validatedData = signUpSchema.parse(formData);

      // Call registration API
      const response = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(validatedData),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Registration failed');
      }

      // Redirect to sign in with success message
      router.push('/signin?registered=true');
    } catch (error) {
      if (error instanceof ZodError) {
        const fieldErrors: Record<string, string> = {};
        error.errors.forEach((err) => {
          if (err.path[0]) {
            fieldErrors[err.path[0] as string] = err.message;
          }
        });
        setErrors(fieldErrors);
      } else if (error instanceof Error) {
        setGeneralError(error.message);
      } else {
        setGeneralError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  const strengthColors = ['bg-neutral-200', 'bg-error-500', 'bg-warning-500', 'bg-success-400', 'bg-success-500'];

  if (authLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="w-8 h-8 border-2 border-cream-200 border-t-navy-500 rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="bg-white rounded-2xl shadow-xl p-8">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="font-heading text-display-sm text-navy-900 mb-2">
          Create your account
        </h1>
        <p className="text-body-md text-neutral-600">
          Start analyzing markets with AI-powered insights
        </p>
      </div>

      {/* General error */}
      {generalError && (
        <div className="mb-6 p-4 bg-error-50 border border-error-200 rounded-lg">
          <p className="text-body-sm text-error-700 flex items-center gap-2">
            <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            {generalError}
          </p>
        </div>
      )}

      {/* Form */}
      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Name field */}
        <div>
          <label htmlFor="name" className="block text-body-sm font-medium text-navy-700 mb-2">
            Full name
          </label>
          <input
            id="name"
            name="name"
            type="text"
            autoComplete="name"
            value={formData.name}
            onChange={handleChange}
            disabled={isSubmitting}
            className={`
              w-full h-12 px-4
              text-body-md text-navy-900
              bg-white border rounded-lg
              transition-all duration-fast
              placeholder:text-neutral-400
              focus:outline-none focus:ring-2 focus:ring-navy-500/20
              disabled:bg-cream-50 disabled:cursor-not-allowed
              ${errors.name 
                ? 'border-error-500 focus:border-error-500' 
                : 'border-border-medium hover:border-border-heavy focus:border-navy-500'
              }
            `}
            placeholder="John Doe"
          />
          {errors.name && (
            <p className="mt-1.5 text-body-sm text-error-500">{errors.name}</p>
          )}
        </div>

        {/* Email field */}
        <div>
          <label htmlFor="email" className="block text-body-sm font-medium text-navy-700 mb-2">
            Email address
          </label>
          <input
            id="email"
            name="email"
            type="email"
            autoComplete="email"
            value={formData.email}
            onChange={handleChange}
            disabled={isSubmitting}
            className={`
              w-full h-12 px-4
              text-body-md text-navy-900
              bg-white border rounded-lg
              transition-all duration-fast
              placeholder:text-neutral-400
              focus:outline-none focus:ring-2 focus:ring-navy-500/20
              disabled:bg-cream-50 disabled:cursor-not-allowed
              ${errors.email 
                ? 'border-error-500 focus:border-error-500' 
                : 'border-border-medium hover:border-border-heavy focus:border-navy-500'
              }
            `}
            placeholder="you@example.com"
          />
          {errors.email && (
            <p className="mt-1.5 text-body-sm text-error-500">{errors.email}</p>
          )}
        </div>

        {/* Password field */}
        <div>
          <label htmlFor="password" className="block text-body-sm font-medium text-navy-700 mb-2">
            Password
          </label>
          <input
            id="password"
            name="password"
            type="password"
            autoComplete="new-password"
            value={formData.password}
            onChange={handleChange}
            disabled={isSubmitting}
            className={`
              w-full h-12 px-4
              text-body-md text-navy-900
              bg-white border rounded-lg
              transition-all duration-fast
              placeholder:text-neutral-400
              focus:outline-none focus:ring-2 focus:ring-navy-500/20
              disabled:bg-cream-50 disabled:cursor-not-allowed
              ${errors.password 
                ? 'border-error-500 focus:border-error-500' 
                : 'border-border-medium hover:border-border-heavy focus:border-navy-500'
              }
            `}
            placeholder="Create a strong password"
          />
          {errors.password && (
            <p className="mt-1.5 text-body-sm text-error-500">{errors.password}</p>
          )}
          
          {/* Password strength indicator */}
          {formData.password && (
            <div className="mt-2">
              <div className="flex gap-1 mb-1">
                {[1, 2, 3, 4].map((level) => (
                  <div
                    key={level}
                    className={`h-1 flex-1 rounded-full transition-colors ${
                      level <= passwordStrength.score ? strengthColors[passwordStrength.score] : 'bg-neutral-200'
                    }`}
                  />
                ))}
              </div>
              <p className="text-caption text-neutral-500">
                Password strength: {passwordStrength.label}
              </p>
            </div>
          )}
        </div>

        {/* Confirm password field */}
        <div>
          <label htmlFor="confirmPassword" className="block text-body-sm font-medium text-navy-700 mb-2">
            Confirm password
          </label>
          <input
            id="confirmPassword"
            name="confirmPassword"
            type="password"
            autoComplete="new-password"
            value={formData.confirmPassword}
            onChange={handleChange}
            disabled={isSubmitting}
            className={`
              w-full h-12 px-4
              text-body-md text-navy-900
              bg-white border rounded-lg
              transition-all duration-fast
              placeholder:text-neutral-400
              focus:outline-none focus:ring-2 focus:ring-navy-500/20
              disabled:bg-cream-50 disabled:cursor-not-allowed
              ${errors.confirmPassword 
                ? 'border-error-500 focus:border-error-500' 
                : 'border-border-medium hover:border-border-heavy focus:border-navy-500'
              }
            `}
            placeholder="Confirm your password"
          />
          {errors.confirmPassword && (
            <p className="mt-1.5 text-body-sm text-error-500">{errors.confirmPassword}</p>
          )}
        </div>

        {/* Terms checkbox */}
        <div>
          <label className="flex items-start gap-3 cursor-pointer">
            <input
              type="checkbox"
              name="acceptTerms"
              checked={formData.acceptTerms}
              onChange={handleChange}
              disabled={isSubmitting}
              className="
                mt-0.5 w-5 h-5
                border-2 border-border-medium rounded
                text-navy-500
                focus:ring-navy-500 focus:ring-offset-0
                cursor-pointer
              "
            />
            <span className="text-body-sm text-neutral-600">
              I agree to the{' '}
              <Link href="/terms" className="text-navy-500 hover:text-navy-700 underline">
                Terms of Service
              </Link>{' '}
              and{' '}
              <Link href="/privacy" className="text-navy-500 hover:text-navy-700 underline">
                Privacy Policy
              </Link>
            </span>
          </label>
          {errors.acceptTerms && (
            <p className="mt-1.5 text-body-sm text-error-500">{errors.acceptTerms}</p>
          )}
        </div>

        {/* Submit button */}
        <button
          type="submit"
          disabled={isSubmitting}
          className="
            w-full h-12
            bg-terra-500 text-white
            font-heading font-medium
            rounded-lg
            transition-all duration-fast
            hover:bg-terra-600 hover:-translate-y-0.5 hover:shadow-terra
            focus:outline-none focus:ring-2 focus:ring-terra-500 focus:ring-offset-2
            disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none disabled:shadow-none
          "
        >
          {isSubmitting ? (
            <span className="flex items-center justify-center gap-2">
              <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Creating account...
            </span>
          ) : (
            'Create account'
          )}
        </button>
      </form>

      {/* Sign in link */}
      <p className="mt-8 text-center text-body-sm text-neutral-600">
        Already have an account?{' '}
        <Link
          href="/signin"
          className="font-medium text-navy-500 hover:text-navy-700 transition-colors"
        >
          Sign in
        </Link>
      </p>
    </div>
  );
}
