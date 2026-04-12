const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

/**
 * IdentityRegistry unit tests — Phase 6.3
 *
 * Run: npx hardhat test
 *
 * Covers:
 *   - Deployment & constructor validation
 *   - registerIdentity: happy path, duplicate NIN revert, non-oracle revert
 *   - grantAccess / revokeAccess
 *   - verifyIdentity: authorised, expired, revoked
 *   - getIpfsCid: access-gated retrieval
 *   - Event emissions
 *   - Gas cost estimation (printed in mocha output)
 */

describe("IdentityRegistry", function () {
  // Test fixtures
  let registry;
  let oracle, subject, verifier, stranger;

  const SAMPLE_HASH_NIN = ethers.keccak256(ethers.toUtf8Bytes("NIN-12345678"));
  const SAMPLE_IPFS_CID = "QmXoypizjW3WknFiJnKLwHCnL72vedxjQkDDP1mXWo6uco";
  const ONE_DAY = 86_400;   // seconds

  beforeEach(async function () {
    [oracle, subject, verifier, stranger] = await ethers.getSigners();
    const Factory = await ethers.getContractFactory("IdentityRegistry");
    registry = await Factory.deploy(oracle.address);
    await registry.waitForDeployment();
  });

  // ── Deployment ─────────────────────────────────────────────────────────────

  describe("Deployment", function () {
    it("sets the oracle address correctly", async function () {
      expect(await registry.oracle()).to.equal(oracle.address);
    });

    it("reverts if oracle is the zero address", async function () {
      const Factory = await ethers.getContractFactory("IdentityRegistry");
      await expect(
        Factory.deploy(ethers.ZeroAddress)
      ).to.be.revertedWith("IdentityRegistry: oracle cannot be zero address");
    });
  });

  // ── registerIdentity ───────────────────────────────────────────────────────

  describe("registerIdentity", function () {
    it("registers an identity and emits IdentityRegistered", async function () {
      const tx = await registry
        .connect(oracle)
        .registerIdentity(subject.address, SAMPLE_HASH_NIN, SAMPLE_IPFS_CID);

      await expect(tx)
        .to.emit(registry, "IdentityRegistered")
        .withArgs(subject.address, SAMPLE_HASH_NIN, SAMPLE_IPFS_CID, await time.latest());

      expect(await registry.isRegistered(subject.address)).to.be.true;
    });

    it("reverts when called by non-oracle", async function () {
      await expect(
        registry
          .connect(stranger)
          .registerIdentity(subject.address, SAMPLE_HASH_NIN, SAMPLE_IPFS_CID)
      ).to.be.revertedWith("IdentityRegistry: caller is not the oracle");
    });

    it("reverts on duplicate NIN (same subject)", async function () {
      await registry
        .connect(oracle)
        .registerIdentity(subject.address, SAMPLE_HASH_NIN, SAMPLE_IPFS_CID);

      await expect(
        registry
          .connect(oracle)
          .registerIdentity(subject.address, SAMPLE_HASH_NIN, SAMPLE_IPFS_CID)
      ).to.be.revertedWith("IdentityRegistry: identity already registered");
    });

    it("reverts for zero subject address", async function () {
      await expect(
        registry
          .connect(oracle)
          .registerIdentity(ethers.ZeroAddress, SAMPLE_HASH_NIN, SAMPLE_IPFS_CID)
      ).to.be.revertedWith("IdentityRegistry: invalid subject address");
    });

    it("reverts for empty IPFS CID", async function () {
      await expect(
        registry
          .connect(oracle)
          .registerIdentity(subject.address, SAMPLE_HASH_NIN, "")
      ).to.be.revertedWith("IdentityRegistry: ipfsCid cannot be empty");
    });

    it("prints gas cost for registerIdentity", async function () {
      const tx = await registry
        .connect(oracle)
        .registerIdentity(subject.address, SAMPLE_HASH_NIN, SAMPLE_IPFS_CID);
      const receipt = await tx.wait();
      console.log(`      ⛽  registerIdentity gas used: ${receipt.gasUsed.toString()}`);
    });
  });

  // ── grantAccess / revokeAccess ─────────────────────────────────────────────

  describe("grantAccess", function () {
    beforeEach(async function () {
      await registry
        .connect(oracle)
        .registerIdentity(subject.address, SAMPLE_HASH_NIN, SAMPLE_IPFS_CID);
    });

    it("grants access and emits AccessGranted", async function () {
      const expiry = (await time.latest()) + ONE_DAY;
      await expect(
        registry.connect(subject).grantAccess(verifier.address, expiry)
      )
        .to.emit(registry, "AccessGranted")
        .withArgs(subject.address, verifier.address, expiry);

      expect(await registry.hasAccess(subject.address, verifier.address)).to.be.true;
    });

    it("reverts when expiry is in the past", async function () {
      const pastExpiry = (await time.latest()) - 1;
      await expect(
        registry.connect(subject).grantAccess(verifier.address, pastExpiry)
      ).to.be.revertedWith("IdentityRegistry: expiry must be in the future");
    });

    it("reverts for zero verifier address", async function () {
      const expiry = (await time.latest()) + ONE_DAY;
      await expect(
        registry.connect(subject).grantAccess(ethers.ZeroAddress, expiry)
      ).to.be.revertedWith("IdentityRegistry: invalid verifier address");
    });

    it("prints gas cost for grantAccess", async function () {
      const expiry = (await time.latest()) + ONE_DAY;
      const tx = await registry.connect(subject).grantAccess(verifier.address, expiry);
      const receipt = await tx.wait();
      console.log(`      ⛽  grantAccess gas used: ${receipt.gasUsed.toString()}`);
    });
  });

  describe("revokeAccess", function () {
    beforeEach(async function () {
      await registry
        .connect(oracle)
        .registerIdentity(subject.address, SAMPLE_HASH_NIN, SAMPLE_IPFS_CID);
      const expiry = (await time.latest()) + ONE_DAY;
      await registry.connect(subject).grantAccess(verifier.address, expiry);
    });

    it("revokes access and emits AccessRevoked", async function () {
      await expect(
        registry.connect(subject).revokeAccess(verifier.address)
      )
        .to.emit(registry, "AccessRevoked")
        .withArgs(subject.address, verifier.address);

      expect(await registry.hasAccess(subject.address, verifier.address)).to.be.false;
    });

    it("reverts when there is no grant to revoke", async function () {
      await expect(
        registry.connect(subject).revokeAccess(stranger.address)
      ).to.be.revertedWith("IdentityRegistry: no access grant to revoke");
    });
  });

  // ── verifyIdentity ─────────────────────────────────────────────────────────

  describe("verifyIdentity", function () {
    beforeEach(async function () {
      await registry
        .connect(oracle)
        .registerIdentity(subject.address, SAMPLE_HASH_NIN, SAMPLE_IPFS_CID);
      const expiry = (await time.latest()) + ONE_DAY;
      await registry.connect(subject).grantAccess(verifier.address, expiry);
    });

    it("returns true and emits AccessLogged for authorised verifier", async function () {
      const result = await registry.connect(verifier).verifyIdentity.staticCall(subject.address);
      expect(result).to.be.true;

      await expect(
        registry.connect(verifier).verifyIdentity(subject.address)
      ).to.emit(registry, "AccessLogged");
    });

    it("reverts when verifier access has expired", async function () {
      await time.increase(ONE_DAY + 1);
      await expect(
        registry.connect(verifier).verifyIdentity(subject.address)
      ).to.be.revertedWith("IdentityRegistry: verifier not authorised or access expired");
    });

    it("reverts for unregistered subject", async function () {
      await expect(
        registry.connect(verifier).verifyIdentity(stranger.address)
      ).to.be.revertedWith("IdentityRegistry: identity not registered");
    });

    it("reverts for unauthorized verifier", async function () {
      await expect(
        registry.connect(stranger).verifyIdentity(subject.address)
      ).to.be.revertedWith("IdentityRegistry: verifier not authorised or access expired");
    });

    it("prints gas cost for verifyIdentity", async function () {
      const tx = await registry.connect(verifier).verifyIdentity(subject.address);
      const receipt = await tx.wait();
      console.log(`      ⛽  verifyIdentity gas used: ${receipt.gasUsed.toString()}`);
    });
  });

  // ── getIpfsCid ─────────────────────────────────────────────────────────────

  describe("getIpfsCid", function () {
    it("returns the IPFS CID for an authorised verifier", async function () {
      await registry
        .connect(oracle)
        .registerIdentity(subject.address, SAMPLE_HASH_NIN, SAMPLE_IPFS_CID);
      const expiry = (await time.latest()) + ONE_DAY;
      await registry.connect(subject).grantAccess(verifier.address, expiry);

      const cid = await registry.connect(verifier).getIpfsCid(subject.address);
      expect(cid).to.equal(SAMPLE_IPFS_CID);
    });
  });
});
