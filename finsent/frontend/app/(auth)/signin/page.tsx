'use client';

// =============================================================================
// SIGN IN PAGE
// =============================================================================

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { useAuth } from '@/lib/auth-context';
import { signInSchema, type SignInInput } from '@/lib/validations/auth';
import { ZodError } from 'zod';

export default function SignInPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { signIn, isAuthenticated, isLoading: authLoading } = useAuth();
  
  const [formData, setFormData] = useState<SignInInput>({
    email: '',
    password: '',
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [generalError, setGeneralError] = useState('');

  // Get callback URL from query params
  const callbackUrl = searchParams.get('callbackUrl') || '/';
  const errorParam = searchParams.get('error');

  // Show error from NextAuth
  useEffect(() => {
    if (errorParam) {
      setGeneralError(
        errorParam === 'CredentialsSignin'
          ? 'Invalid email or password'
          : 'An error occurred. Please try again.'
      );
    }
  }, [errorParam]);

  // Redirect if already authenticated
  useEffect(() => {
    if (isAuthenticated && !authLoading) {
      router.push(callbackUrl);
    }
  }, [isAuthenticated, authLoading, router, callbackUrl]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
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
      const validatedData = signInSchema.parse(formData);

      // Attempt sign in
      const result = await signIn(validatedData.email, validatedData.password);

      if (result.success) {
        router.push(callbackUrl);
      } else {
        setGeneralError(result.error || 'Sign in failed. Please try again.');
      }
    } catch (error) {
      if (error instanceof ZodError) {
        const fieldErrors: Record<string, string> = {};
        error.errors.forEach((err) => {
          if (err.path[0]) {
            fieldErrors[err.path[0] as string] = err.message;
          }
        });
        setErrors(fieldErrors);
      } else {
        setGeneralError('An unexpected error occurred. Please try again.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

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
          Welcome back
        </h1>
        <p className="text-body-md text-neutral-600">
          Sign in to your account to continue
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
          <div className="flex items-center justify-between mb-2">
            <label htmlFor="password" className="block text-body-sm font-medium text-navy-700">
              Password
            </label>
            <Link
              href="/forgot-password"
              className="text-body-sm font-medium text-navy-500 hover:text-navy-700 transition-colors"
            >
              Forgot password?
            </Link>
          </div>
          <input
            id="password"
            name="password"
            type="password"
            autoComplete="current-password"
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
            placeholder="Enter your password"
          />
          {errors.password && (
            <p className="mt-1.5 text-body-sm text-error-500">{errors.password}</p>
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
              Signing in...
            </span>
          ) : (
            'Sign in'
          )}
        </button>
      </form>

      {/* Divider */}
      <div className="my-6 flex items-center">
        <div className="flex-1 border-t border-border-light" />
        <span className="px-4 text-body-sm text-neutral-500">or</span>
        <div className="flex-1 border-t border-border-light" />
      </div>

      {/* OAuth buttons (placeholder - enable in auth.ts) */}
      <div className="space-y-3">
        <button
          type="button"
          disabled
          className="
            w-full h-12 px-4
            flex items-center justify-center gap-3
            bg-white border border-border-medium
            text-navy-700 font-medium
            rounded-lg
            transition-all duration-fast
            hover:bg-cream-50 hover:border-border-heavy
            disabled:opacity-50 disabled:cursor-not-allowed
          "
        >
          <svg className="w-5 h-5" viewBox="0 0 24 24">
            <path fill="#4285F4" d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
            <path fill="#34A853" d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
            <path fill="#FBBC05" d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
            <path fill="#EA4335" d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
          </svg>
          Continue with Google
        </button>
      </div>

      {/* Sign up link */}
      <p className="mt-8 text-center text-body-sm text-neutral-600">
        Don&apos;t have an account?{' '}
        <Link
          href="/signup"
          className="font-medium text-navy-500 hover:text-navy-700 transition-colors"
        >
          Create one
        </Link>
      </p>
    </div>
  );
}
