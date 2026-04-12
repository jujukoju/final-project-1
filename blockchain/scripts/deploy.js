/**
 * blockchain/scripts/deploy.js — Phase 6.4
 *
 * Deploys IdentityRegistry to the configured network.
 * Writes deployed address + ABI to blockchain/deployed.json for use by other services.
 *
 * Usage:
 *   npx hardhat run scripts/deploy.js --network localhost
 *   npx hardhat run scripts/deploy.js --network sepolia
 */

const { ethers, network } = require("hardhat");
const fs   = require("fs");
const path = require("path");

async function main() {
  const [deployer] = await ethers.getSigners();

  console.log("─".repeat(55));
  console.log("  Deploying IdentityRegistry");
  console.log(`  Network  : ${network.name}`);
  console.log(`  Deployer : ${deployer.address}`);
  console.log(
    `  Balance  : ${ethers.formatEther(await ethers.provider.getBalance(deployer.address))} ETH`
  );
  console.log("─".repeat(55));

  // The oracle is the deployer for the localhost/testnet prototype.
  // In production, this would be the monitored NIMC oracle account.
  const oracleAddress = deployer.address;

  const Factory  = await ethers.getContractFactory("IdentityRegistry");
  const contract = await Factory.deploy(oracleAddress);
  await contract.waitForDeployment();

  const deployedAddress = await contract.getAddress();
  console.log(`  Contract deployed to: ${deployedAddress}`);

  // ── Persist address + ABI ─────────────────────────────────────────────────
  const artifact = await artifacts.readArtifact("IdentityRegistry");
  const outPath  = path.join(__dirname, "..", "deployed.json");

  fs.writeFileSync(
    outPath,
    JSON.stringify({
      network:     network.name,
      address:     deployedAddress,
      oracle:      oracleAddress,
      deployedAt:  new Date().toISOString(),
      abi:         artifact.abi,
    }, null, 2),
    "utf8"
  );

  console.log(`  ABI + address saved → ${outPath}`);
  console.log("─".repeat(55));
}

main()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error(err);
    process.exit(1);
  });
